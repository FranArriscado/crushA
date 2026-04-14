@echo off
setlocal

if "%~1"=="" (
    echo Drag a MAT file onto this BAT file to plot it and save a JPG.
    pause
    exit /b 1
)

set "MAT_FILE=%~f1"
set "SCRIPT_DIR=%~dp0"

if /I not "%~x1"==".mat" (
    echo Expected a .mat file, got: %~nx1
    pause
    exit /b 1
)

if not exist "%MAT_FILE%" (
    echo File not found: %MAT_FILE%
    pause
    exit /b 1
)

set "MAT_FILE=%MAT_FILE:\=/%"
set "SCRIPT_DIR=%SCRIPT_DIR:\=/%"

echo Plotting %MAT_FILE%
echo.

matlab -batch "addpath('%SCRIPT_DIR%'); plot_partition_mesh('%MAT_FILE%','PlotMode','2D','SaveJpg',true);"

if errorlevel 1 (
    echo.
    echo MATLAB reported an error while plotting the file.
    pause
    exit /b 1
)

echo.
echo Plot created successfully.
pause
