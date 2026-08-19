@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The local .venv was not found. Run setup_windows_pycharm.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" quick_run.py %*
if errorlevel 1 (
  echo Quick test failed.
  pause
  exit /b 1
)
pause
