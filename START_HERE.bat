@echo off
REM ---------------------------------------------------------------------
REM  Nairobi flood susceptibility - one-click setup and launch (Windows)
REM
REM  Double-click this file, or run it from a command prompt. It finds a
REM  suitable Python, then hands over to scripts\bootstrap.py, which
REM  installs dependencies, extracts the rasters, trains if needed, and
REM  starts the API and the app.
REM
REM  Python 3.12 or newer is required: the pinned rasterio, pyproj,
REM  xgboost, shap and numpy publish no wheels for anything older.
REM ---------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "PYEXE="

REM The py launcher is the most reliable way to pick a specific version.
for %%V in (3.13 3.12) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>&1 && set "PYEXE=py -%%V"
    )
)

REM Otherwise fall back to whatever `python` is on PATH, if it is new enough.
if not defined PYEXE (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" >nul 2>&1 && set "PYEXE=python"
)

if not defined PYEXE (
    echo.
    echo   No Python 3.12 or newer was found.
    echo.
    echo   If you have Anaconda, open an Anaconda Prompt and run:
    echo.
    echo       conda create -n flood python=3.13 -y
    echo       conda activate flood
    echo       python scripts\bootstrap.py --no-venv
    echo.
    echo   Otherwise install Python 3.13 from https://www.python.org/downloads/
    echo   and tick "Add python.exe to PATH" during setup, then run this file again.
    echo.
    pause
    exit /b 1
)

echo Using %PYEXE%
%PYEXE% scripts\bootstrap.py %*

if errorlevel 1 (
    echo.
    echo   Setup stopped. The reason is printed above.
    pause
    exit /b 1
)
endlocal
