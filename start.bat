@echo off
setlocal
cd /d "%~dp0"

if not exist .env (
  copy .env.example .env >nul
  echo [INFO] Created .env from .env.example
)

start "Simple NotebookLM API" cmd /k "python -m uvicorn src.interfaces.api:app --reload --port 8000"
timeout /t 2 /nobreak >nul
start "Simple NotebookLM UI" cmd /k "python -m streamlit run src.interfaces.ui"

echo API: http://localhost:8000/docs
echo UI : http://localhost:8501
endlocal
