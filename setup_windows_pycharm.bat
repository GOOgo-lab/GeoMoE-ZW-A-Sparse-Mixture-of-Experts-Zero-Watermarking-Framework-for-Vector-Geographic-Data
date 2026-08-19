@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Locating Python 3.10 or newer...
set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 set "PY_CMD=py -3"
if not defined PY_CMD (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ and select "Add Python to PATH".
    pause
    exit /b 1
  )
  set "PY_CMD=python"
)
%PY_CMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo Python 3.10 or newer is required.
  pause
  exit /b 1
)

echo [2/4] Creating the local virtual environment .venv...
%PY_CMD% -m venv .venv
if errorlevel 1 goto :failed

echo [3/4] Installing the quick-test dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements-quick.txt
if errorlevel 1 goto :failed

echo [4/4] Running the GeoMoE-ZW quick test...
".venv\Scripts\python.exe" quick_run.py
if errorlevel 1 goto :failed

echo.
echo Setup completed successfully.
echo In PyCharm select: .venv\Scripts\python.exe
pause
exit /b 0

:failed
echo.
echo Setup failed. Review the message above and confirm network access and Python version.
pause
exit /b 1
