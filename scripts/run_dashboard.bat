@echo off
REM Football Prediction Lab — Run Local Dashboard
setlocal enabledelayedexpansion

REM Resolve repository root from BAT script directory (%~dp0 is the scripts directory)
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

echo ==============================================================================
echo FOOTBALL PREDICTION MODEL LAB — LOCAL RESEARCH DASHBOARD
echo Repository Root: %CD%
echo Launching: streamlit run src\dashboard\app.py
echo ==============================================================================

streamlit run "src\dashboard\app.py" --server.headless false

pause
