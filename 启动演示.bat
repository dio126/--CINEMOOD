@echo off
rem Double-click launcher for the Streamlit demo app
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   Starting Streamlit demo (movie-sentiment)
echo   Browser will open at http://localhost:8501
echo   Close this window to stop the server.
echo ============================================
rem "echo." pipes an empty line to auto-skip Streamlit's first-run email prompt
echo. | .venv\Scripts\python.exe -m streamlit run app\app.py
pause
