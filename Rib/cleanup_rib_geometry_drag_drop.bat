@echo off
setlocal EnableExtensions

set "THIS_DIR=%~dp0"
if "%THIS_DIR:~-1%"=="\" set "THIS_DIR=%THIS_DIR:~0,-1%"
for %%I in ("%THIS_DIR%\..") do set "ROOT_DIR=%%~fI"

set "PYTHON_EXE=C:\Users\Arriscado\envs\crushenv\Scripts\python.exe"
set "CLEANUP_SCRIPT=%ROOT_DIR%\cleanup_partition_geometry_branched.py"

if not exist "%PYTHON_EXE%" (
    echo Python executable not found:
    echo   %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

if not exist "%CLEANUP_SCRIPT%" (
    echo Rib cleanup script not found:
    echo   %CLEANUP_SCRIPT%
    echo.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Drag and drop one or more rib partition .mat files onto this BAT file.
    echo.
    echo It will run:
    echo   %CLEANUP_SCRIPT% --resample-uniform
    echo.
    echo Output files are saved next to the input as:
    echo   ^<input^>_CU.mat
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo Rib Geometry Cleanup
echo Python : %PYTHON_EXE%
echo Script : %CLEANUP_SCRIPT%
echo ============================================================
echo.

set "FAILED=0"

:process_next
if "%~1"=="" goto done

echo ------------------------------------------------------------
echo Input: %~1
set "OUTPUT_FILE=%~dpn1_CU%~x1"
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
