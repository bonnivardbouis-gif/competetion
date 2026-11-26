@echo off
cd /d "%~dp0"
if not exist "runtime\python.exe" (
    echo [ERROR] runtime\python.exe not found. Package is incomplete.
    pause
    exit /b 1
)
"runtime\python.exe" main.py %*
echo.
pause