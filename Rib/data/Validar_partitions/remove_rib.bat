@echo off
setlocal EnableExtensions

set "THIS_DIR=%~dp0"
if "%THIS_DIR:~-1%"=="\" set "THIS_DIR=%THIS_DIR:~0,-1%"

set "PYTHON_EXE=C:\Users\Arriscado\envs\crushenv\Scripts\python.exe"
set "REMOVE_SCRIPT=%THIS_DIR%\remove_rib_from_geometry.py"

if not exist "%PYTHON_EXE%" (
    echo Python executable not found:
    echo   %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

if not exist "%REMOVE_SCRIPT%" (
    echo Rib removal script not found:
    echo   %REMOVE_SCRIPT%
    echo.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Drag and drop one or more ribbed .mat files onto this BAT file.
    echo.
    echo It will run:
    echo   %REMOVE_SCRIPT%
    echo.
    echo Output files are saved next to the input as:
    echo   ^<input^>_ribremoved.mat
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo Rib Removal
echo Python : %PYTHON_EXE%
echo Script : %REMOVE_SCRIPT%
echo ============================================================
echo.

set "FAILED=0"

:process_next
if "%~1"=="" goto done

echo ------------------------------------------------------------
echo Input : %~1
set "OUTPUT_FILE=%~dpn1_ribremoved%~x1"
echo Output: %OUTPUT_FILE%
"%PYTHON_EXE%" "%REMOVE_SCRIPT%" "%~1" -o "%OUTPUT_FILE%"
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
