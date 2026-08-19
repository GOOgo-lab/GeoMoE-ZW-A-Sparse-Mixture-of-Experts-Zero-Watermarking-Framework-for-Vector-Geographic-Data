# GeoMoE-ZW user guide

## Public quick-test mode

### Input

No external file is required. `quick_run.py` generates a deterministic
32-dimensional registered descriptor, a perturbed query descriptor, ten dummy
expert projections and a 256-bit copyright watermark.

Options:

```bash
python quick_run.py --seed 20260318 --attack-strength 0.035
```

- `--seed`: controls all generated values; the default is `20260318`.
- `--attack-strength`: standard deviation of the synthetic query perturbation;
  the default is `0.035`.

### Processing

The script constructs `[query, registered, absolute difference, safe ratio]`
to obtain 128 values, profiles ten candidate experts over 12 simulated attack
directions, selects the registration set, constructs one XOR zero-watermark per
registered expert, routes to Top-2 experts, and evaluates recovered watermarks.

### Output

The command writes one JSON object to standard output. Important fields are:

- `registered_experts`: experts retained by the capability rule.
- `descriptor_dim`: must be 128.
- `selected_experts`: Top-2 routed experts.
- `per_expert_nc`: recovery NC for each active expert.
- `best_nc`: maximum recovery NC.
- `ber`: bit error rate of the selected watermark.
- `accepted`: whether `best_nc >= 0.80`.

## Paper implementation mode

The final paper-facing modules are in `rb_afl_system/paper`:

- `descriptor.py`: deterministic 128-dimensional paired descriptor.
- `router.py`: three-layer RouterNet and delta=0.02 multi-label objective.
- `registry.py`: capability selection and serializable registration records.
- `protocol.py`: 256-bit registration and Top-2 verification.

Preprocessed samples are directories containing:

```text
sample/
|-- grid.npy       float array [4, 256, 256]
|-- tokens.npz     object tokens and optional validity mask
|-- graph.npz      node features, adjacency and optional node mask
`-- metadata.json  optional vector statistics
```

The registered and query samples must use the same preprocessing convention.
Model checkpoints and dataset paths are supplied through local configuration;
do not commit them to Git.

## Paper attack protocol

The exact executable configurations are:

- `configs/paper_single_attacks_12.json`: 12 distinct single-attack families.
- `configs/paper_mixed_attacks_10.json`: ordered manuscript combinations M1-M10.

`RUN_IN_PYCHARM.py` checks both counts even in `ACTION="quick"`. In a formal
`ACTION="all"` run, it builds the single-attack dataset, creates identity-level
70/20/10 train/validation/test splits, extends them with M1-M10, trains on the
single-attack training split, and evaluates both the single and mixed held-out
test splits. Use `ACTION="prepare_mixed"`, `"evaluate_single"`, or
`"evaluate_mixed"` to resume a specific stage.

During formal evaluation the code loads `WATERMARK_PATH`, resizes and binarizes
it to 256 bits, applies `ARNOLD_ITERATIONS`, and registers
`zero_watermark = copyright_bits XOR base_feature_bits` independently for every
selected expert and identity. RouterNet selects Top-2 experts before recovery;
the candidate with maximum NC determines the final result. Cross-identity
registration/query pairs are evaluated separately to calculate FMR@0.80.

## Expected failures

- `ModuleNotFoundError: torch`: install the full optional dependencies.
- Missing Mapshaper: follow `INSTALL_MAPSHAPER.md` or use internal attacks only.
- Missing `grid.npy`: build the preprocessed dataset before formal evaluation.
- Router output mismatch: the RouterNet output dimension must equal the number
  of registered experts in the registry.
