@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Launch the GUI immediately; do not download or import PyTorch here.
rem GPU runtime setup is an explicit one-time operation:
rem     scripts\setup_cuda_runtime.bat
set "PYTHONFAULTHANDLER=1"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "VPY=%CD%\.venv_live38\Scripts\python.exe"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "OPENBLAS_NUM_THREADS=2"
set "NUMEXPR_NUM_THREADS=2"

if not exist "%VPY%" (
    echo LVIMOT dedicated CARLA 0.9.13 Python 3.8 environment is missing.
    echo Setting up the environment using scripts\setup_live_carla_env.bat...
    call scripts\setup_live_carla_env.bat
    if errorlevel 1 exit /b 1
)
if not exist "lvimot\output\live_gui.py" (
    echo [ERROR] lvimot\output\live_gui.py is missing.
    exit /b 2
)
if not exist "lvimot\output\pipeline_inspector.py" (
    echo [ERROR] lvimot\output\pipeline_inspector.py is missing.
    exit /b 2
)

rem The backend auto-selects GPU/CPU; checking CUDA before the GUI was
rem repeatedly causing long startup delays on Windows.
if exist ".cuda_runtime\torch\__init__.py" (
    set "PYTHONPATH=%CD%\.cuda_runtime;%PYTHONPATH%"
)

"%VPY%" -u lvimot\output\live_gui.py
exit /b %ERRORLEVEL%
