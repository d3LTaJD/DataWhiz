@echo off
echo ========================================
echo    DataWhiz - Electron Installation
echo ========================================
echo.

echo [1/4] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed!
    echo Please install Node.js from https://nodejs.org/
    echo Then run this script again.
    pause
    exit /b 1
)
echo ✓ Node.js is installed

echo.
echo [2/4] Installing Electron dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo ✓ Dependencies installed successfully

echo.
echo [3/4] Installing Python dependencies...
py -m pip install flask flask-cors pandas numpy scikit-learn plotly openpyxl
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies!
    pause
    exit /b 1
)
echo ✓ Python dependencies installed

echo.
echo [4/4] Creating necessary directories...
if not exist "uploads" mkdir uploads
if not exist "assets" mkdir assets
echo ✓ Directories created

echo.
echo ========================================
echo    Installation Complete!
echo ========================================
echo.
echo To start DataWhiz:
echo   npm start
echo.
echo To start in development mode:
echo   npm run dev
echo.
pause
