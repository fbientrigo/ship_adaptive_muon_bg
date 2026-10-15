@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

"%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%scripts\run_afterms_d9_5_nightly.py" start %*
if errorlevel 1 (
    echo.
    echo run_d9_5_night.cmd failed with exit code %errorlevel%.
    pause
)

endlocal
