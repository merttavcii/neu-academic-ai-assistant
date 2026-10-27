@echo off
title Dental AI Assistant Launcher

echo ================================
echo Starting Dental AI Assistant...
echo ================================

REM Backend başlat
start "Backend - FastAPI" cmd /k "cd /d C:\Users\Mesut\Desktop\Bitirme\dental-ai-assistant\backend && venv\Scripts\activate && uvicorn main:app --reload --host 0.0.0.0 --port 8010"

REM Frontend başlat
start "Frontend - React Vite" cmd /k "cd /d C:\Users\Mesut\Desktop\Bitirme\dental-ai-assistant\frontend && npm run dev"

REM Tarayıcıyı aç
timeout /t 5 >nul
start http://localhost:5173

echo Project started.
pause