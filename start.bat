@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python scripts\prepare.py
if errorlevel 1 exit /b 1
cd web
call npm install
if errorlevel 1 exit /b 1
call npm run build
if errorlevel 1 exit /b 1
cd ..
start "AI Studio" cmd /c "timeout /t 5 >nul & start http://localhost:8000"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
