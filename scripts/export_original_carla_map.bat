@echo off
setlocal
cd /d "%~dp0.."
set "VPY=%CD%\.venv_live38\Scripts\python.exe"
if not exist "%VPY%" (
  echo [ERROR] LVIMOT Python environment is missing. Run scripts\setup_live_carla_env.bat first.
  pause
  exit /b 2
)
"%VPY%" -u tools\export_original_carla_map.py
if errorlevel 1 (
  echo.
  echo Make sure CARLA is running, the town is Town10HD_Opt, and you finished one LVIMOT recording.
)
echo.
pause
