@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

rem Rebuild the exact saved LVIMOT-vs-CARLA full-scene viewer. No LVIMOT rerun.
rem If the matching independent 360 PLY is absent, CARLA 0.9.13 Town10HD_Opt
rem must be running. The script renders that scene from the saved GT route.
set "PYTHON_CMD=python"
if exist ".venv_live38\Scripts\python.exe" set "PYTHON_CMD=.venv_live38\Scripts\python.exe"

if not exist "outputs_carla\live\offline_4d_map.ply" (
  echo [ERROR] Missing outputs_carla\live\offline_4d_map.ply
  pause
  exit /b 2
)
if not exist "outputs_carla\live\offline_4d_scene.json" (
  echo [ERROR] Missing outputs_carla\live\offline_4d_scene.json
  pause
  exit /b 2
)
if not exist "outputs_carla\live\live_results.json" (
  echo [ERROR] Missing outputs_carla\live\live_results.json
  pause
  exit /b 2
)

echo [COMPARISON] Loading LVIMOT complete estimated map.
echo [COMPARISON] Using full CARLA 360 rendered scene, NOT the OpenDRIVE road-only reference.
echo [COMPARISON] If missing, CARLA must be running and LVIMOT must be stopped.
"%PYTHON_CMD%" -u tools\regenerate_offline_html.py --output-dir "outputs_carla\live" --capture-if-missing --require-full-reference
if errorlevel 1 (
  echo [ERROR] No valid full CARLA/LVIMOT comparison was generated.
  echo Keep CARLA 0.9.13 Town10HD_Opt running and retry.
  pause
  exit /b 1
)
start "" "outputs_carla\live\offline_4d_map.html"
exit /b 0
