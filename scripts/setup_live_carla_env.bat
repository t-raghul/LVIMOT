@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

set "VPY=%CD%\.venv_live38\Scripts\python.exe"
if not exist "%VPY%" (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Install Python 3.8 x64 with the Windows py launcher.
        exit /b 2
    )
    py -3.8 -m venv ".venv_live38"
    if errorlevel 1 (
        echo [ERROR] Python 3.8 x64 is required for CARLA 0.9.13 on Windows.
        exit /b 2
    )
)

"%VPY%" -m pip install "pip<25" wheel setuptools
if errorlevel 1 exit /b 10

rem Preserve a working Torch installation. Install CPU Torch only if absent.
"%VPY%" -c "import torch; import torchvision" >nul 2>nul
if errorlevel 1 (
    echo Installing verified CPU PyTorch for Python 3.8.
    "%VPY%" -m pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 exit /b 11
)

"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 12

"%VPY%" -c "import sys, carla, cv2, numpy, scipy, torch, ultralytics; assert sys.version_info[:2] == (3,8); print('LVIMOT environment ready; Python', sys.version.split()[0], 'Torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
if errorlevel 1 exit /b 13

exit /b 0
