@echo off
echo ======================================================================
echo Launching Predictive Workforce Intelligence Decision System...
echo ======================================================================

cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run src\app\app.py
pause
