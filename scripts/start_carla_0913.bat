@echo off
setlocal EnableExtensions
rem Usage: scripts\start_carla_0913.bat "C:\path\to\CARLA_0.9.13\WindowsNoEditor"
rem Or set CARLA_INSTALL in your environment.
set "CARLA_DIR=%CARLA_INSTALL%"
if not "%~1"=="" set "CARLA_DIR=%~1"
if "%CARLA_DIR%"=="" (
    echo [ERROR] Set CARLA_INSTALL or pass your WindowsNoEditor directory.
    exit /b 2
)
if not exist "%CARLA_DIR%\CarlaUE4.exe" (
    echo [ERROR] CarlaUE4.exe not found in "%CARLA_DIR%"
    exit /b 3
)
cd /d "%CARLA_DIR%"
start "" "CarlaUE4.exe"
