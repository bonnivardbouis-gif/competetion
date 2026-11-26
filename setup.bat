@echo off
cd /d "%~dp0"
echo This package is fully self-contained.
echo runtime\ already bundles Python 3.13 + all dependencies, so you normally do NOT need this script.
echo.
echo   Web UI : double-click run_webui.bat
echo   CLI    : double-click run.bat
echo.
echo If runtime\ is damaged, re-extract the original llm_sci_agent.zip.
pause