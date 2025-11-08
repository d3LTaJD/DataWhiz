@echo off
echo ========================================
echo    DataWhiz - Professional Analytics
echo ========================================
echo.

echo Starting DataWhiz...
echo.

echo [1/2] Starting Python backend...
start "DataWhiz Backend" py backend/run.py

echo [2/2] Starting Electron frontend...
echo Please wait for the backend to start...
timeout /t 3 /nobreak >nul

npx electron .

echo.
echo DataWhiz is now running!
echo.
pause