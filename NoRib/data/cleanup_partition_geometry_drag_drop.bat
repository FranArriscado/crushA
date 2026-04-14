@echo off
setlocal EnableExtensions

set "THIS_DIR=%~dp0"
if "%THIS_DIR:~-1%"=="\" set "THIS_DIR=%THIS_DIR:~0,-1%"

set "VENV_DIR=C:\Users\Arriscado\envs\crushenv"
set "ACTIVATE_BAT=%VENV_DIR%\Scripts\activate.bat"
set "CLEANUP_SCRIPT=%THIS_DIR%\cleanup_partition_geometry.py"

if not exist "%ACTIVATE_BAT%" (
    echo Virtual environment activation script not found:
    echo   %ACTIVATE_BAT%
    echo.
    pause
    exit /b 1
)

if not exist "%CLEANUP_SCRIPT%" (
    echo Cleanup script not found:
    echo   %CLEANUP_SCRIPT%
    echo.
    pause
    exit /b 1
)

call "%ACTIVATE_BAT%"
if errorlevel 1 (
    echo Failed to activate virtual environment:
    echo   %VENV_DIR%
    echo.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Drag and drop one or more closed-loop partition .mat files onto this BAT file.
    echo.
    echo It will run:
    echo   python %CLEANUP_SCRIPT% --resample-uniform
    echo.
    echo Output files are saved next to the input as:
    echo   ^<input^>_uniform.mat
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo Partition Geometry Cleanup
echo Venv   : %VENV_DIR%
echo Script : %CLEANUP_SCRIPT%
echo Mode   : Uniform node spacing enabled
echo ============================================================
echo.

set "FAILED=0"

:process_next
if "%~1"=="" goto done

echo ------------------------------------------------------------
echo Input : %~1
set "OUTPUT_FILE=%~dpn1_uniform%~x1"
echo Output: %OUTPUT_FILE%
python "%CLEANUP_SCRIPT%" "%~1" --resample-uniform -o "%OUTPUT_FILE%"
if errorlevel 1 (
    echo [FAILED] %~1
    set "FAILED=1"
) else (
    echo [OK] %~1
)
echo.
shift
goto process_next

:done
echo ============================================================
if "%FAILED%"=="0" (
    echo Finished successfully.
) else (
    echo Finished with errors.
)
echo ============================================================
echo.
pause
exit /b %FAILED%
