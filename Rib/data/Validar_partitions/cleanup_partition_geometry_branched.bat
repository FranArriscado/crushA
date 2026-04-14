@echo off
setlocal EnableExtensions

set "THIS_DIR=%~dp0"
if "%THIS_DIR:~-1%"=="\" set "THIS_DIR=%THIS_DIR:~0,-1%"

set "PYTHON_EXE=C:\Users\Arriscado\envs\crushenv\Scripts\python.exe"
set "CLEANUP_SCRIPT=%THIS_DIR%\cleanup_partition_geometry_branched.py"

if not exist "%PYTHON_EXE%" (
    echo Python executable not found:
    echo   %PYTHON_EXE%
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

if "%~1"=="" (
    echo Drag and drop one or more branched/open partition .mat files onto this BAT file.
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
echo Python : %PYTHON_EXE%
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
"%PYTHON_EXE%" "%CLEANUP_SCRIPT%" "%~1" --resample-uniform -o "%OUTPUT_FILE%"
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
