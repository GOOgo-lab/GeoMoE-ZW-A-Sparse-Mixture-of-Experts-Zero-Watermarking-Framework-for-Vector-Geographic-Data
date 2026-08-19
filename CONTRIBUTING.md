# Contributing

This open-source repository accompanies an academic manuscript. Before opening a pull
request, please:

1. Keep manuscript constants in `configs/paper_geomoe_zw.json` unchanged unless
   the change is explicitly documented as a new experimental variant.
2. Run `python quick_run.py` and the relevant tests.
3. Do not commit datasets, shapefiles, checkpoints, generated results, logs, or
   machine-specific absolute paths.
4. Describe the dataset split, random seed, attack settings and hardware for
   every result-affecting change.
