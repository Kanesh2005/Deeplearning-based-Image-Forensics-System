@echo off
title SPECTRA::FORENSICS Web App Server
cd /d "%~dp0"
echo ============================================================
echo         SPECTRA::FORENSICS WEB APPLICATION LAUNCHER         
echo ============================================================
echo Using virtual environment Python: .\venv\Scripts\python.exe
echo Starting Flask Server on http://127.0.0.1:5000 ...
echo ============================================================
.\venv\Scripts\python.exe app.py
pause
