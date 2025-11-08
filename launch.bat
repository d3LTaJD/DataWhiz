@echo off
cd /d "%~dp0"
echo ========================================
echo    DataWhiz - Professional Analytics
echo ========================================
echo.

echo Starting DataWhiz...
echo.

echo [1/2] Starting Python backend...
start "DataWhiz Backend" cmd /c "py backend/app.py & pause"

echo [2/2] Starting Electron frontend...
echo Please wait for the backend to start...
timeout /t 5 /nobreak >nul

echo Starting Electron application...
npx electron .

echo.
echo DataWhiz is now running!
echo.
pause
