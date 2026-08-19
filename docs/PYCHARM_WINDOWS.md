# Running GeoMoE-ZW on Windows with PyCharm

## Fastest setup

1. Install Python 3.10 (64-bit) and PyCharm.
2. Clone or download the public repository and keep all extracted source files
   in one folder. Do not open the transport ZIP directly in PyCharm.
3. Double-click `setup_windows_pycharm.bat`.
4. Wait until the window prints `Setup completed successfully` and the JSON
   quick-test result contains `"accepted": true`.
5. In PyCharm, choose **File > Open** and select the repository folder.
6. Open **File > Settings > Project > Python Interpreter**.
7. Select **Add Interpreter > Add Local Interpreter > Existing** and choose:

   ```text
   <repository>\.venv\Scripts\python.exe
   ```

8. Open `quick_run.py`, right-click in the editor, and select
   **Run 'quick_run'**.

Alternatively, open `RUN_IN_PYCHARM.py` and run it. Leave `ACTION="quick"` for
the immediate test. For a formal run, edit `SOURCE_ROOT`, `WATERMARK_PATH` and
`OUTPUT_ROOT`, then set `ACTION="all"`. The complete action runs the 12 single
attacks, creates identity-disjoint 70/20/10 splits, builds mixed attacks M1-M10,
trains the selected experts, evaluates single and mixed held-out sets, and
exports separate Excel summaries.

## PyCharm run configuration

If a configuration must be created manually:

- Configuration type: `Python`
- Script path: `<repository>\quick_run.py`
- Working directory: `<repository>`
- Python interpreter: `<repository>\.venv\Scripts\python.exe`
- Parameters: leave empty, or use
  `--seed 20260318 --attack-strength 0.035`

The quick test requires only NumPy and runs on CPU. It should finish within a
few seconds on a typical Windows computer.

## Installing the full research environment

Open the PyCharm terminal after selecting `.venv`, then run:

```bat
python -m pip install -e ".[full,test]"
```

For an NVIDIA GPU, install a PyTorch build compatible with the local CUDA and
driver versions by following the official PyTorch installation selector. The
full experiments also require the vector dataset, trained checkpoints and
Mapshaper. The quick test does not require any of them.

## Common Windows issues

- `python is not recognized`: reinstall Python and enable **Add Python to PATH**.
- `No module named numpy`: run
  `.venv\Scripts\python.exe -m pip install -r requirements-quick.txt`.
- PyCharm uses another interpreter: select the repository-local `.venv` again.
- Chinese or space-containing folder names cause an external utility failure:
  move the repository to a short path such as `D:\GeoMoE-ZW` for formal runs.
- Mapshaper is not found: follow `INSTALL_MAPSHAPER.md` and restart PyCharm.
