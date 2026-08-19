# Reproducibility statement

## What is publicly reproducible without restricted assets

`quick_run.py` and `examples/quick_test.py` provide a deterministic synthetic
test for the complete registration, paired-descriptor, sparse-routing,
watermark-recovery and NC-decision control flow. They require only Python and
NumPy and are executed automatically by GitHub Actions.

## What is required to reproduce manuscript tables and figures

The numerical values in the manuscript require the exact vector dataset split,
trained expert checkpoints, attack instances and random seed. The repository
contains the source code and configuration definitions, but it does not
redistribute third-party vector data or large binary checkpoints. Users must
obtain vector data under the original provider's terms and generate local
preprocessed samples. Any public release of the authors' trained checkpoints
should be attached as a versioned GitHub Release or deposited in a persistent
research repository, with a checksum and license, rather than committed to Git.

## Paper constants

`configs/paper_geomoe_zw.json` is the source of truth for the final manuscript:

- 256 x 256 four-channel raster input;
- 24-dimensional object tokens;
- ten candidate experts and 256-dimensional embeddings;
- 12 attack directions;
- registration threshold NC=0.80 and at least three registered experts;
- 128-dimensional paired routing descriptor;
- RouterNet hidden dimensions 128 and 64;
- delta=0.02 oracle margin and Top-2 activation;
- 256-bit watermark and NC=0.80 authentication threshold.

The final executable attack lists are stored separately in
`configs/paper_single_attacks_12.json` and
`configs/paper_mixed_attacks_10.json`. Their counts and M1-M10 ordering are
validated by the one-click quick action and GitHub Actions.

Formal output includes per-sample selected experts, recovered-watermark
XNOR-based bitwise NC and BER. Under this definition, NC = 1 - BER. The output
also includes the acceptance decision, attack-level Mean/Min NC and RPR, and
cross-identity maximum NC and FMR. For compatibility with the original research
code, mismatched watermark/feature bit arrays are truncated to their shared
prefix before XOR and metric calculation.

Historical V15-V19 scripts are included for provenance. Their exploratory
settings must not override the final paper configuration when reproducing the
reported protocol.
