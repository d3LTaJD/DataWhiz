@echo off
echo Testing DataWhiz Backend Startup...
echo.

echo [1] Checking Python...
py --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found! Please install Python 3.11+
    pause
    exit /b 1
)

echo.
echo [2] Checking backend folder...
if not exist "backend\run.py" (
    echo ERROR: Backend folder not found!
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo.
echo [3] Starting backend...
echo Backend should start on http://localhost:5000
echo Press Ctrl+C to stop it
echo.
cd backend
py run.py

