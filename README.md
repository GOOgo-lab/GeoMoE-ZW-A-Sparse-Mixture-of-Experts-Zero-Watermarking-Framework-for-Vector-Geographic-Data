# GeoMoE-ZW

[![Quick check](https://github.com/YOUR_GITHUB_USERNAME/GeoMoE-ZW/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/GeoMoE-ZW/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official source-code repository for the manuscript:

> **GeoMoE-ZW: An Adaptive Sparse Mixture-of-Experts Zero-Watermarking Method for Robust Authentication of Vector Geographic Data**

Corresponding author: Lanqing Wang (`3540897163@qq.com`).

## Purpose

GeoMoE-ZW is a zero-watermarking framework for robust copyright authentication
of vector geographic data. It combines four-channel raster, object-token and
topology-graph representations with a ten-model candidate expert pool. A
128-dimensional paired descriptor drives a three-layer RouterNet, which selects
the Top-2 registered experts without requiring an attack label at inference.
Each selected expert independently recovers a 256-bit watermark, and the
highest XNOR-based bitwise normalized correlation (NC) is used for
authentication. Matching bits contribute 1 and mismatching bits contribute 0,
so NC = 1 - BER. Legacy arrays with unequal lengths are compared on their
shared prefix, matching the original implementation.

The software contains the model definitions, data preprocessing and attack
operators, expert training/evaluation utilities, capability profiling,
zero-watermark registration/recovery, sparse routing, mixed-attack evaluation,
and a dependency-light synthetic quick test.

## Repository layout

```text
GeoMoE-ZW/
|-- rb_afl_system/          Core Python package
|   |-- data/               Vector I/O, preprocessing and attacks
|   |-- models/             CNN, ResNet-SE, token, graph and fusion experts
|   |-- losses/             Training objectives
|   |-- watermark/          Bit generation, XOR registration and metrics
|   |-- router/             Geometry/topology descriptors
|   `-- paper/              Final paper-aligned protocol
|-- configs/                Model and paper experiment configurations
|-- examples/               Quick-test entry point
|-- tests/                  Automated checks
|-- docs/                   User guide and reproducibility notes
|-- quick_run.py            CPU-only synthetic end-to-end test
|-- pyproject.toml          Installation metadata
`-- LICENSE                 MIT open-source license
```

No ZIP/RAR archive, private dataset, trained checkpoint, credential, or
machine-specific output is stored in the repository.

## Quick test (recommended first run)

Requirements: Python 3.10 or newer and NumPy. No GPU, vector dataset, PyTorch,
or trained checkpoint is required.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/GeoMoE-ZW.git
cd GeoMoE-ZW
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# Linux or macOS
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python quick_run.py
```

Equivalent example-file command:

```bash
python examples/quick_test.py
```

Expected output includes:

```json
{
  "mode": "synthetic_cpu_smoke_test",
  "candidate_experts": 10,
  "descriptor_dim": 128,
  "top_k": 2,
  "best_nc": 0.992218,
  "threshold": 0.8,
  "accepted": true
}
```

The exact displayed structure should match, and `accepted` should be `true`
with the default seed. This quick test validates control flow and interfaces;
it is not evidence for the numerical results reported in the manuscript.

### Windows and PyCharm

On Windows, double-click `setup_windows_pycharm.bat`. It creates a local
`.venv`, installs NumPy, and runs the quick test. Then open the repository in
PyCharm and select `.venv\Scripts\python.exe` as the project interpreter. Full
screens-by-menu instructions are provided in
[`docs/PYCHARM_WINDOWS.md`](docs/PYCHARM_WINDOWS.md).

For a one-file PyCharm workflow, open `RUN_IN_PYCHARM.py`. Its default
`ACTION="quick"` runs immediately. For a formal experiment, edit the three
Windows paths at the top, set `ACTION="all"`, and run the same file. Other
actions allow resuming only preparation, training, evaluation, or XLSX export.
The `all` workflow now validates and builds all 12 single attacks from
`configs/paper_single_attacks_12.json` and all 10 ordered mixed attacks M1-M10
from `configs/paper_mixed_attacks_10.json`, then evaluates both held-out sets.

## Full installation

Formal model training and evaluation require the optional scientific stack:

```bash
python -m pip install -e ".[full,test]"
```

Computational environment used in the manuscript:

- Python 3.10
- PyTorch 2.x
- CUDA 12.x
- NVIDIA RTX 4090 with 24 GB VRAM
- Mapshaper for selected external vector attacks
- approximately 32 GB system RAM recommended

Mapshaper installation details are in [INSTALL_MAPSHAPER.md](INSTALL_MAPSHAPER.md).

## Formal workflow

1. Prepare vector layers and build the four-channel grid, 24-dimensional object
   tokens and topology graph.
2. Train the ten candidate experts using
   `configs/model_suite_reduced_candidate_pool_v17_5.json` and the constants in
   `configs/paper_geomoe_zw.json`.
3. Evaluate 12 attack directions and select experts using worst-direction
   NC >= 0.80, while retaining at least three experts.
4. Register one 256-bit XOR zero-watermark per selected expert.
5. Train RouterNet with paired 128-dimensional descriptors, delta=0.02
   multi-label targets, AdamW, and Top-2 routing.
6. Evaluate single and mixed attacks and report Mean NC, Min NC, NC<0.80,
   RPR@0.80, and FMR@0.80.

The formal one-click evaluator reads the copyright image, converts it to 256
binary bits, applies the configured Arnold scrambling, creates one XOR
zero-watermark per registered identity and selected expert, trains RouterNet on
the single-attack training split, and performs Top-2 recovery. Mixed attacks use
the same trained router without mixed-attack retraining. Results are written to
`results/single/GeoMoE_ZW_paper_results.xlsx` and
`results/mixed/GeoMoE_ZW_paper_results.xlsx`.

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for inputs, outputs and options,
and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the relationship
between the public test and the manuscript experiments.

## Tests

Dependency-light public check:

```bash
python quick_run.py
python -m compileall -q rb_afl_system quick_run.py
```

After installing the full dependencies:

```bash
python -m pytest
python tests/smoke_core.py
```

GitHub Actions automatically executes the dependency-light check on every push
and pull request.

## Data and checkpoints

The repository does not redistribute third-party geographic datasets or large
trained checkpoints. The quick test supplies deterministic synthetic features
and dummy expert projections so that anonymous users can verify installation
and execute the complete registration-routing-recovery control flow. See the
reproducibility document for the justified limitations and required directory
structure for formal experiments.

## Citation

If this repository contributes to your work, cite the accompanying manuscript.
Citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

The code is released under the [MIT License](LICENSE) and can be downloaded
anonymously from the public GitHub repository.

## Contact

Lanqing Wang: `3540897163@qq.com`

Before making the repository public, replace every occurrence of
`YOUR_GITHUB_USERNAME` with the actual GitHub account name.
