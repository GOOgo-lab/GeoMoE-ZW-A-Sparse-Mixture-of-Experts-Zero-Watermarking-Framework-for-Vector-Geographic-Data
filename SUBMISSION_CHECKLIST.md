# Computers & Geosciences resubmission checklist

Complete every item before resubmission:

- [ ] Create a **public** GitHub repository named `GeoMoE-ZW`.
- [ ] Upload the extracted individual files and folders, not a ZIP/RAR/7z file.
- [ ] Replace all `YOUR_GITHUB_USERNAME` placeholders with the real account.
- [ ] Confirm that `README.md` is visible on the repository landing page.
- [ ] Confirm that the MIT `LICENSE` is detected by GitHub.
- [ ] Confirm that the `quick-check` GitHub Action passes.
- [ ] In a signed-out/incognito browser, open the repository and download it.
- [ ] From a fresh clone, run `python -m pip install -e .` and
      `python quick_run.py`.
- [ ] On a clean Windows/PyCharm setup, run `setup_windows_pycharm.bat`, select
      `.venv\Scripts\python.exe`, and run `quick_run.py` successfully.
- [ ] Confirm that no dataset, owner identifier, private watermark, token,
      credential, checkpoint or unpublished result was committed.
- [ ] Copy the final text from `COMPUTER_CODE_AVAILABILITY.md` to a manuscript
      section titled exactly **Computer Code Availability**.
- [ ] Put that section at the end of the manuscript and use the real public URL.
- [ ] Keep the main-text word count within the journal limit; exclude Abstract,
      Keywords, Highlights, References, Captions and Appendices as instructed by
      the editorial letter.
