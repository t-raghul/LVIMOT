@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
set "VPY=%CD%\.venv_live38\Scripts\python.exe"
if not exist "%VPY%" (
    echo [ERROR] Run scripts\setup_live_carla_env.bat first.
    exit /b 2
)
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo No NVIDIA driver detected. LVIMOT will use its CPU runtime.
    exit /b 0
)
echo Preparing the isolated GPU runtime for Python 3.8. CPU packages stay available.
echo The first download can exceed 2 GB. Close LVIMOT before updating an existing GPU runtime.
"%VPY%" -m pip install --target "%CD%\.cuda_runtime" --upgrade --no-deps --timeout 120 --retries 3 "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 exit /b 10
"%VPY%" -m pip install --target "%CD%\.cuda_runtime" --upgrade --no-deps "threadpoolctl==3.5.0"
if errorlevel 1 exit /b 11
set "PYTHONPATH=%CD%\.cuda_runtime;%PYTHONPATH%"
"%VPY%" -c "import torch; assert torch.cuda.is_available(), 'GPU runtime installed but CUDA is unavailable'; print('GPU ready:',torch.cuda.get_device_name(0),'PyTorch',torch.__version__)"
exit /b %ERRORLEVEL%
