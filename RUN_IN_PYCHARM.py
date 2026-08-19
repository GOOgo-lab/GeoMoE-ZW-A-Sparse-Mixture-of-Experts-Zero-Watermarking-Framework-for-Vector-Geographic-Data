"""One-click PyCharm entry point for GeoMoE-ZW on Windows.

Usage
-----
1. Open this repository folder in PyCharm.
2. For a formal run, edit SOURCE_ROOT, WATERMARK_PATH and OUTPUT_ROOT below.
3. Set ACTION to quick, prepare, prepare_mixed, train, evaluate_single,
   evaluate_mixed, evaluate, xlsx or all.
4. Right-click this file and choose "Run 'RUN_IN_PYCHARM'".

The ``quick`` action needs only NumPy. The other actions use the real vector
data, expert models and evaluation pipeline described in the manuscript.
"""
from __future__ import annotations

import json
import runpy
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================================
# REQUIRED FOR FORMAL RUNS: use raw Windows strings, for example r"D:\data"
# These paths are ignored when ACTION = "quick".
# ============================================================================

SOURCE_ROOT = r"F:\1\data\shp"
WATERMARK_PATH = r"F:\1\1.png"
OUTPUT_ROOT = r"F:\1\runs\paper_run"


# ============================================================================
# RUN SETTINGS
# ============================================================================

# quick: dependency-light synthetic end-to-end validation (recommended first)
# prepare: build all 12 single attacks and create identity-disjoint splits
# prepare_mixed: extend the prepared dataset with the 10 paper mixed attacks
# train: reuse prepared samples and train the selected candidate experts
# evaluate_single: evaluate the 12 single attacks on the held-out test split
# evaluate_mixed: evaluate the 10 mixed attacks on the held-out test split
# evaluate: run both single-attack and mixed-attack evaluation
# xlsx: rebuild the Excel summary from existing evaluation results
# all: prepare -> train -> evaluate -> Excel summary
ACTION = "evaluate"

# Empty means all ten paper candidate experts. Enter names to debug a subset.
SELECTED_MODELS: list[str] = []

GRID_SIZE = 256
EPOCHS = 800
BATCH_SIZE_OVERRIDE: int | None = None
NUM_WORKERS = 0  # Keep zero for Windows/PyCharm stability.
DEVICE = "auto"  # auto / cpu / cuda
BIT_LENGTH = 256
THRESHOLD_MODE = "mean"
COPYRIGHT_IMAGE_THRESHOLD = 128
ARNOLD_ITERATIONS = 5
NC_THRESHOLD = 0.80
BER_THRESHOLD = 0.20
SEED = 20260318

# Use "auto" for automatic discovery, or provide mapshaper.cmd's full path.
MAPSHAPER_BIN = "auto"
AUTO_INSTALL_MAPSHAPER = False
REQUIRE_MAPSHAPER = False


PAPER_TOP10 = [
    "component_relation_transformer_topology_v15",
    "geogrid_graph",
    "resnet_se_spectralD",
    "geovecformer_robust_unique",
    "cnn_fc",
    "proxyanchor_gridcnn_unique_v15",
    "geovecformer_deepD_supcon",
    "cnn_deepD_supcon",
    "geotoken_base_unique",
    "geotoken_only",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_npm() -> str | None:
    candidates = [
        shutil.which("npm.cmd"),
        shutil.which("npm"),
        r"C:\Program Files\nodejs\npm.cmd",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    return None


def _find_mapshaper() -> str | None:
    import os

    candidates = [shutil.which("mapshaper.cmd"), shutil.which("mapshaper")]
    for variable in ("APPDATA", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base:
            candidates.append(str(Path(base) / "npm" / "mapshaper.cmd"))
    npm = _find_npm()
    if npm:
        try:
            prefix = subprocess.run(
                [npm, "config", "get", "prefix"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            if prefix:
                candidates.extend([str(Path(prefix) / "mapshaper.cmd"), str(Path(prefix) / "mapshaper")])
        except (OSError, subprocess.SubprocessError):
            pass
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    return None


def _resolve_mapshaper() -> str | None:
    if MAPSHAPER_BIN.strip().lower() != "auto":
        path = Path(MAPSHAPER_BIN).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MAPSHAPER_BIN does not exist: {path}")
        return str(path)
    found = _find_mapshaper()
    if found:
        return found
    npm = _find_npm()
    if AUTO_INSTALL_MAPSHAPER and npm:
        print("[setup] Mapshaper was not found; running npm install -g mapshaper...", flush=True)
        result = subprocess.run([npm, "install", "-g", "mapshaper"], check=False)
        if result.returncode == 0:
            found = _find_mapshaper()
            if found:
                return found
    if REQUIRE_MAPSHAPER:
        raise FileNotFoundError("Mapshaper is required but was not found. See INSTALL_MAPSHAPER.md.")
    print("[warning] External Mapshaper attacks will be skipped; internal attacks remain enabled.", flush=True)
    return None


def _selected_names() -> list[str]:
    selected = SELECTED_MODELS or PAPER_TOP10
    unknown = sorted(set(selected) - set(PAPER_TOP10))
    if unknown:
        raise ValueError(f"Unknown SELECTED_MODELS entries: {unknown}")
    return selected


def _paths() -> tuple[Path, Path, Path, Path]:
    output = Path(OUTPUT_ROOT).expanduser().resolve()
    return output, output / "prepared_dataset", output / "models", output / "results"


def _extended_paths() -> dict[str, Path]:
    output, prepared, models, results = _paths()
    return {
        "output": output,
        "single_dataset": prepared,
        "single_splits": output / "single_splits",
        "mixed_dataset": output / "mixed_dataset",
        "mixed_splits": output / "mixed_splits",
        "models": models,
        "results": results,
    }


def _validate_paths(action: str) -> None:
    if action in {"all", "prepare", "prepare_mixed"}:
        source = Path(SOURCE_ROOT).expanduser().resolve()
        if not source.is_dir():
            raise NotADirectoryError(f"SOURCE_ROOT does not exist: {source}")
        if not list(source.rglob("*.shp")):
            raise ValueError(f"No .shp file was found under SOURCE_ROOT: {source}")
    if action in {"all", "evaluate", "evaluate_single", "evaluate_mixed"}:
        watermark = Path(WATERMARK_PATH).expanduser().resolve()
        if not watermark.is_file():
            raise FileNotFoundError(f"WATERMARK_PATH does not exist: {watermark}")
    paths = _extended_paths()
    prepared, models = paths["single_dataset"], paths["models"]
    if action in {"train", "evaluate", "evaluate_single", "evaluate_mixed", "prepare_mixed"} and not (prepared / "metadata.csv").is_file():
        raise FileNotFoundError(f"Prepared dataset is missing: {prepared / 'metadata.csv'}")
    if action in {"evaluate", "evaluate_single", "evaluate_mixed"}:
        missing = [name for name in _selected_names() if not (models / name / "best.pt").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing best.pt checkpoints for: {missing}")


def _model_definitions() -> tuple[dict, dict, list[dict]]:
    suite = _read_json(PROJECT_ROOT / "configs" / "model_suite_reduced_candidate_pool_v17_5.json")
    by_name = {str(item["name"]): item for item in suite["models"]}
    missing = [name for name in _selected_names() if name not in by_name]
    if missing:
        raise ValueError(f"Candidate configuration is missing models: {missing}")
    return dict(suite["common_train"]), dict(suite["common_eval"]), [by_name[name] for name in _selected_names()]


def run_quick() -> None:
    single = _read_json(PROJECT_ROOT / "configs" / "paper_single_attacks_12.json")["attacks"]
    mixed = _read_json(PROJECT_ROOT / "configs" / "paper_mixed_attacks_10.json")["mixed_attacks"]
    single_types = [str(item["attack_type"]) for item in single]
    mixed_ids = [str(item["id"]) for item in mixed]
    if len(single) != 12 or len(set(single_types)) != 12:
        raise RuntimeError(f"Expected 12 unique single attacks, got {single_types}")
    if mixed_ids != [f"M{i}" for i in range(1, 11)]:
        raise RuntimeError(f"Expected ordered mixed attacks M1-M10, got {mixed_ids}")
    print(f"[config] Single attacks: {len(single)} ({', '.join(single_types)})")
    print(f"[config] Mixed attacks: {len(mixed)} ({', '.join(mixed_ids)})")
    print("[quick] Running the CPU-only synthetic GeoMoE-ZW validation...", flush=True)
    runpy.run_path(str(PROJECT_ROOT / "quick_run.py"), run_name="__main__")


def prepare() -> Path:
    from rb_afl_system.data.dataset.build_dataset import build_identity_dataset

    _, prepared, _, _ = _paths()
    config = _read_json(PROJECT_ROOT / "configs" / "paper_single_attacks_12.json")
    mapshaper_bin = _resolve_mapshaper()
    config.update(
        source_root=str(Path(SOURCE_ROOT).expanduser().resolve()),
        output_root=str(prepared),
        grid_size=GRID_SIZE,
        mapshaper_bin=mapshaper_bin or "mapshaper",
        seed=SEED,
    )
    if mapshaper_bin is None:
        config["attacks"] = [
            attack for attack in config.get("attacks", [])
            if str(attack.get("engine", "internal")).lower() != "mapshaper"
        ]
    print("\n[1/4] Building multimodal samples from SHP files...", flush=True)
    summary = build_identity_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    from rb_afl_system.scripts.split_dataset_formal_V14 import split_dataset_formal

    paths = _extended_paths()
    split_info = split_dataset_formal(
        dataset_root=prepared,
        output_root=paths["single_splits"],
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10,
        seed=SEED,
        min_identities_per_split=1,
        check_paths=True,
    )
    print(json.dumps(split_info, ensure_ascii=False, indent=2, default=str))
    return prepared


def prepare_mixed() -> Path:
    from rb_afl_system.scripts.extend_dataset_mixed_attacks_V17_2 import extend_mixed

    paths = _extended_paths()
    mapshaper_bin = _resolve_mapshaper() or "mapshaper"
    print("\n[2/5] Building the 10 ordered paper mixed attacks M1-M10...", flush=True)
    summary = extend_mixed(
        base_dataset_root=paths["single_dataset"],
        base_split_root=paths["single_splits"],
        source_root=Path(SOURCE_ROOT).expanduser().resolve(),
        output_dataset_root=paths["mixed_dataset"],
        output_split_root=paths["mixed_splits"],
        mixed_config_path=PROJECT_ROOT / "configs" / "paper_mixed_attacks_10.json",
        force=True,
        seed=SEED,
        mapshaper_bin=mapshaper_bin,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return paths["mixed_dataset"]


def train() -> Path:
    from rb_afl_system.engine.adversarial_trainer import train_adversarial

    paths = _extended_paths()
    prepared, models = paths["single_splits"] / "train", paths["models"]
    common_train, _, definitions = _model_definitions()
    common_train.update(
        dataset_root=str(prepared),
        epochs=EPOCHS,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        seed=SEED,
    )
    if BATCH_SIZE_OVERRIDE is not None:
        common_train["batch_size"] = int(BATCH_SIZE_OVERRIDE)
    print("\n[2/4] Training the selected paper candidate experts...", flush=True)
    for index, item in enumerate(definitions, start=1):
        name = str(item["name"])
        config = {**common_train, **dict(item.get("train", {}))}
        config.update(dataset_root=str(prepared), output_dir=str(models / name))
        if BATCH_SIZE_OVERRIDE is not None:
            config["batch_size"] = int(BATCH_SIZE_OVERRIDE)
        print(f"[{index}/{len(definitions)}] Training {name}", flush=True)
        train_adversarial(config)
    return models


def evaluate(dataset_kind: str = "single") -> Path:
    from rb_afl_system.engine.paper_moe_evaluator import evaluate_paper_moe, train_paper_router

    paths = _extended_paths()
    models, results = paths["models"], paths["results"] / dataset_kind
    if dataset_kind == "single":
        prepared = paths["single_splits"] / "test"
    elif dataset_kind == "mixed":
        prepared = paths["mixed_splits"] / "test_mixed_only"
    else:
        raise ValueError(f"Unknown dataset_kind: {dataset_kind}")
    if not (prepared / "metadata.csv").is_file():
        raise FileNotFoundError(f"Evaluation metadata is missing: {prepared / 'metadata.csv'}")
    router_dir = paths["results"] / "paper_router"
    router_checkpoint = router_dir / "paper_router.pt"
    if not router_checkpoint.is_file():
        print("\n[router] Training the paper RouterNet from the single-attack training split...", flush=True)
        train_paper_router({
            "train_dataset_root": str(paths["single_splits"] / "train"),
            "model_root": str(models), "model_names": _selected_names(),
            "copyright_image_path": str(Path(WATERMARK_PATH).expanduser().resolve()),
            "copyright_image_threshold": COPYRIGHT_IMAGE_THRESHOLD,
            "arnold_iterations": ARNOLD_ITERATIONS, "bit_length": BIT_LENGTH,
            "threshold_mode": THRESHOLD_MODE, "nc_threshold": NC_THRESHOLD,
            "router_epochs": 800, "router_lr": 0.002, "router_weight_decay": 0.0001,
            "oracle_margin": 0.02, "device": DEVICE, "seed": SEED,
            "output_dir": str(router_dir),
        })
    print(f"\n[4/5] Running paper GeoMoE-ZW Top-2 evaluation on {dataset_kind} attacks...", flush=True)
    summary = evaluate_paper_moe({
        "dataset_root": str(prepared), "model_root": str(models),
        "router_checkpoint": str(router_checkpoint),
        "copyright_image_path": str(Path(WATERMARK_PATH).expanduser().resolve()),
        "copyright_image_threshold": COPYRIGHT_IMAGE_THRESHOLD,
        "arnold_iterations": ARNOLD_ITERATIONS, "bit_length": BIT_LENGTH,
        "threshold_mode": THRESHOLD_MODE, "nc_threshold": NC_THRESHOLD,
        "top_k": 2, "device": DEVICE, "output_dir": str(results),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    workbook = results / "GeoMoE_ZW_paper_results.xlsx"
    print(f"[complete] Paper results: {workbook}", flush=True)
    return workbook


def rebuild_xlsx() -> Path:
    results = _extended_paths()["results"]
    workbooks = [results / kind / "GeoMoE_ZW_paper_results.xlsx" for kind in ("single", "mixed")]
    missing = [str(path) for path in workbooks if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Paper result workbooks do not exist; run evaluation first: {missing}")
    for workbook in workbooks:
        print(f"[complete] Existing paper results: {workbook}", flush=True)
    return workbooks[-1]


def main() -> None:
    action = ACTION.strip().lower()
    allowed = {"quick", "all", "prepare", "prepare_mixed", "train", "evaluate_single", "evaluate_mixed", "evaluate", "xlsx"}
    if action not in allowed:
        raise ValueError(f"ACTION must be one of: {sorted(allowed)}")
    if action == "quick":
        run_quick()
        return
    _validate_paths(action)
    output, _, _, _ = _paths()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output root: {output}")
    print(f"Action: {action}")
    print(f"Models: {_selected_names()}")
    if action in {"all", "prepare"}:
        prepare()
    if action in {"all", "prepare_mixed"}:
        prepare_mixed()
    if action in {"all", "train"}:
        train()
    if action in {"all", "evaluate", "evaluate_single"}:
        evaluate("single")
    if action in {"all", "evaluate", "evaluate_mixed"}:
        evaluate("mixed")
    if action == "xlsx":
        rebuild_xlsx()
    print("\nGeoMoE-ZW workflow completed.", flush=True)


if __name__ == "__main__":
    main()
