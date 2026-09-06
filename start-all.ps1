# ==============================================================================
# Frigate NVR + Live Webcam Stream All-in-One Startup Script
# ==============================================================================

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting Frigate NVR + Live Webcam Surveillance System  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Start Docker Container (Frigate)
Write-Host "`n[1/2] Starting Frigate Docker Container..." -ForegroundColor Yellow
docker compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Failed to start Docker container. Please ensure Docker Desktop is running." -ForegroundColor Red
    exit 1
}
Write-Host "[+] Frigate container is running!" -ForegroundColor Green

# 2. Launch Python Webcam Hub (with Multi-Camera Selector)
Write-Host "`n[2/2] Launching Camera Selector Hub..." -ForegroundColor Yellow
$portInUse = Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue

if ($portInUse) {
    Write-Host "[*] Camera hub is already running on port 8080." -ForegroundColor Green
} else {
    Write-Host "[*] Spawning webcam_server.py in a dedicated terminal window..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; python webcam_server.py"
    Start-Sleep -Seconds 2
}

# Open browser to camera selector
Start-Process "http://localhost:8080"

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  All Services Started Successfully!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Frigate Web UI:        http://localhost:5000" -ForegroundColor White
Write-Host "  Webcam Live Preview:   http://localhost:8080" -ForegroundColor White
Write-Host "  go2rtc Streams UI:     http://localhost:1984" -ForegroundColor White
Write-Host "==========================================================`n" -ForegroundColor Green
