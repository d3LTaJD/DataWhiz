Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   DataWhiz - Professional Analytics" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting DataWhiz..." -ForegroundColor Green
Write-Host ""

Write-Host "[1/2] Starting Python backend..." -ForegroundColor Yellow
Start-Process -FilePath "py" -ArgumentList "backend/app.py" -WindowStyle Normal

Write-Host "[2/2] Starting Electron frontend..." -ForegroundColor Yellow
Write-Host "Please wait for the backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "Starting Electron application..." -ForegroundColor Green
npx electron .

Write-Host ""
Write-Host "DataWhiz is now running!" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to continue"
