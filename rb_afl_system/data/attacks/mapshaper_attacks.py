#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mapshaper-based topology-aware attacks.

This module invokes the real mapshaper CLI. It does not simulate mapshaper.
Install mapshaper with Node.js, for example:
    npm install -g mapshaper

V06 notes
---------
- Windows .CMD/.BAT invocations keep percent arguments unchanged, e.g. ``80%``.
- Some mixed-geometry GeoJSON inputs are split by Mapshaper into multiple output
  layers such as ``output1.geojson`` and ``output2.geojson`` instead of the
  requested ``output.geojson``. V06 detects these layer outputs and merges them
  back into a single GeoDataFrame.
- Missing-output errors include command/stdout/stderr/temp-dir listing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

try:
    import geopandas as gpd
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Mapshaper wrapper requires geopandas and pandas") from exc


@dataclass
class MapshaperRunResult:
    command: list[str]
    stdout: str
    stderr: str
    returncode: int
    output_exists: bool
    output_paths: list[str] = field(default_factory=list)


def require_mapshaper(binary: str = "mapshaper") -> str:
    """Return a callable mapshaper executable or raise a clear error."""
    raw = str(binary).strip()
    if not raw:
        raw = "mapshaper"
    p = Path(raw)
    if p.is_file():
        return str(p)
    exe = shutil.which(raw)
    if exe is None:
        raise RuntimeError(
            "mapshaper executable was not found. Install Node.js and then run "
            "`npm install -g mapshaper`, or set mapshaper_bin to the full path "
            "of mapshaper/mapshaper.cmd."
        )
    return exe


def _is_windows_batch(exe: str) -> bool:
    return os.name == "nt" and Path(exe).suffix.lower() in {".cmd", ".bat"}


def _run_subprocess(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    cmd = [str(x) for x in cmd]
    if cmd and _is_windows_batch(cmd[0]):
        # Do not escape percent signs here. On this user's Windows + PowerShell
        # setup, passing 80%% makes Mapshaper receive the literal invalid value
        # "80%%". Passing 80% is correct.
        command_line = subprocess.list2cmdline(cmd)
        return subprocess.run(
            command_line,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def mapshaper_version(binary: str = "mapshaper") -> str:
    exe = require_mapshaper(binary)
    result = _run_subprocess([exe, "-v"])
    text = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(f"mapshaper -v failed: {text}")
    return text


def _mapshaper_output_candidates(out: Path) -> list[Path]:
    """Return expected output plus Mapshaper split-layer outputs.

    Mapshaper can split mixed-geometry GeoJSON into output1.geojson,
    output2.geojson, ... even when output.geojson is requested. This function
    collects both cases in deterministic order.
    """
    out = Path(out)
    candidates: list[Path] = []
    if out.is_file():
        candidates.append(out)

    stem = out.stem
    suffix = out.suffix
    for p in sorted(out.parent.glob(f"{stem}*{suffix}")):
        if p.is_file() and p not in candidates:
            candidates.append(p)
    return candidates


def _format_failure(cmd: Sequence[str], result: subprocess.CompletedProcess[str], out: Path, tmpdir: Path | None = None) -> str:
    listing = ""
    if tmpdir is not None and tmpdir.exists():
        listing = "\nTemp dir files:\n" + "\n".join(str(p) for p in sorted(tmpdir.rglob("*")))
    return (
        "mapshaper command did not create expected output\n"
        f"command: {' '.join(str(x) for x in cmd)}\n"
        f"expected output: {out}\n"
        f"returncode: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
        f"{listing}"
    )


def _success_result(cmd: list[str], result: subprocess.CompletedProcess[str], output_paths: list[Path]) -> MapshaperRunResult:
    return MapshaperRunResult(
        command=cmd,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        output_exists=bool(output_paths),
        output_paths=[str(p) for p in output_paths],
    )


def run_mapshaper(
    input_vector: str | Path,
    output_vector: str | Path,
    commands: List[str],
    binary: str = "mapshaper",
    tmpdir_for_diagnostics: Path | None = None,
) -> MapshaperRunResult:
    exe = require_mapshaper(binary)
    inp = Path(input_vector)
    out = Path(output_vector)
    if not inp.is_file():
        raise FileNotFoundError(str(inp))
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [exe, str(inp)] + [str(x) for x in commands] + ["-o", str(out)]
    result = _run_subprocess(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            "mapshaper command failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    output_paths = _mapshaper_output_candidates(out)
    if output_paths:
        return _success_result(cmd, result, output_paths)

    # Fallback for mapshaper versions that infer output format poorly.
    fallback_cmd = [exe, str(inp)] + [str(x) for x in commands] + ["-o", "format=geojson", str(out)]
    fallback_result = _run_subprocess(fallback_cmd)
    if fallback_result.returncode != 0:
        raise RuntimeError(
            "mapshaper fallback command failed\n"
            f"command: {' '.join(fallback_cmd)}\n"
            f"stdout:\n{fallback_result.stdout}\n"
            f"stderr:\n{fallback_result.stderr}"
        )
    fallback_output_paths = _mapshaper_output_candidates(out)
    if fallback_output_paths:
        return _success_result(fallback_cmd, fallback_result, fallback_output_paths)

    raise RuntimeError(_format_failure(fallback_cmd, fallback_result, out, tmpdir=tmpdir_for_diagnostics))


def _read_and_merge_outputs(paths: list[str]) -> gpd.GeoDataFrame:
    gdfs: list[gpd.GeoDataFrame] = []
    crs = None
    for item in paths:
        path = Path(item)
        if not path.is_file():
            continue
        part = gpd.read_file(path)
        if part.empty:
            continue
        if crs is None:
            crs = part.crs
        gdfs.append(part)
    if not gdfs:
        raise RuntimeError(f"mapshaper produced no readable non-empty output layers: {paths}")
    if len(gdfs) == 1:
        return gdfs[0]
    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True, sort=False), crs=crs)
    if "geometry" in merged.columns:
        merged = merged.set_geometry("geometry")
    return merged


def mapshaper_simplify_gdf(
    gdf: gpd.GeoDataFrame,
    keep_percent: float,
    method: str = "weighted",
    clean: bool = True,
    keep_shapes: bool = True,
    binary: str = "mapshaper",
) -> gpd.GeoDataFrame:
    """Simplify a GeoDataFrame through real mapshaper and read it back."""
    if not (0.0 < keep_percent <= 100.0):
        raise ValueError(f"keep_percent must be in (0, 100], got {keep_percent}")
    with tempfile.TemporaryDirectory(prefix="rbafl_mapshaper_") as tmp:
        tmpdir = Path(tmp)
        inp = tmpdir / "input.geojson"
        out = tmpdir / "output.geojson"
        gdf.to_file(inp, driver="GeoJSON")
        simplify_args = ["-simplify", method, f"{float(keep_percent):g}%"]
        if keep_shapes:
            simplify_args.append("keep-shapes")
        commands = simplify_args
        if clean:
            commands = commands + ["-clean"]
        run_result = run_mapshaper(inp, out, commands=commands, binary=binary, tmpdir_for_diagnostics=tmpdir)
        return _read_and_merge_outputs(run_result.output_paths)
