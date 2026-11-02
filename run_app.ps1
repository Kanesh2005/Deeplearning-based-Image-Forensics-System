# SPECTRA::FORENSICS PowerShell Launcher
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "        SPECTRA::FORENSICS WEB APPLICATION LAUNCHER         " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Using Virtual Environment Python: .\venv\Scripts\python.exe" -ForegroundColor Green
Write-Host "Starting Flask Server at http://127.0.0.1:5000 ..." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

$env:PYTHONUNBUFFERED="1"
.\venv\Scripts\python.exe app.py
