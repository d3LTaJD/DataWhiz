const { app, BrowserWindow, Menu, dialog, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let splashWindow;
let splashStartTime = null;
let backendProcess = null;
let appIsQuiting = false;

// Development/Debug mode detection
const isDevelopment = process.argv.includes('--dev') || !app.isPackaged;

// Log mode on startup
if (isDevelopment) {
  console.log('='.repeat(50));
  console.log('DataWhiz - DEVELOPMENT MODE');
  console.log('='.repeat(50));
  console.log('Debug logging: ENABLED');
  console.log('DevTools: Will open with --dev flag');
  console.log('='.repeat(50));
}

function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            // Handle new file
          }
        },
        {
          label: 'Open',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            // Handle open file
          }
        },
        { type: 'separator' },
        {
          label: 'Exit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo', label: 'Undo' },
        { role: 'redo', label: 'Redo' },
        { type: 'separator' },
        { role: 'cut', label: 'Cut' },
        { role: 'copy', label: 'Copy' },
        { role: 'paste', label: 'Paste' },
        { role: 'selectAll', label: 'Select All' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload', label: 'Reload' },
        { role: 'forceReload', label: 'Force Reload' },
        { type: 'separator' },
        { role: 'resetZoom', label: 'Actual Size' },
        { role: 'zoomIn', label: 'Zoom In' },
        { role: 'zoomOut', label: 'Zoom Out' },
        { type: 'separator' },
        { role: 'togglefullscreen', label: 'Toggle Full Screen' }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize', label: 'Minimize' },
        { role: 'close', label: 'Close' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About DataWhiz',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About DataWhiz',
              message: 'DataWhiz - Professional Data Analytics Platform',
              detail: 'A modern desktop application for data science and analytics.\n\n' +
                      'Version: 1.0.0\n' +
                      'Build Date: 2025\n\n' +
                      'Technology Stack:\n' +
                      '• Frontend: Electron + HTML/CSS/JavaScript\n' +
                      '• Backend: Python Flask API\n' +
                      '• Data Processing: Pandas, NumPy\n' +
                      '• Visualization: Plotly\n' +
                      '• Machine Learning: Scikit-learn\n\n' +
                      'License: MIT License\n' +
                      'Built for professional data analytics.',
              buttons: ['OK']
            });
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

function createSplashWindow() {
  // Create splash screen window
  const iconPath = path.join(__dirname, 'mind-map.png');
  splashWindow = new BrowserWindow({
    width: 450,
    height: 580,
    frame: false,
    transparent: false,
    alwaysOnTop: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    resizable: false,
    show: false,
    title: 'DataWhiz - Loading',
    backgroundColor: '#0d1117',
    icon: iconPath,
    skipTaskbar: true
  });
  
  // Ensure splash window icon is set
  splashWindow.setIcon(iconPath);

  // Load splash screen
  splashWindow.loadFile('splash.html');

  // Show splash screen when ready
  splashWindow.once('ready-to-show', () => {
    splashWindow.show();
    splashStartTime = Date.now(); // Track when splash screen is shown
  });
}

function createWindow() {
  // Create the browser window
  const iconPath = path.join(__dirname, 'mind-map.png');
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true,
      spellcheck: false
    },
    icon: iconPath,
    show: false,
    title: 'DataWhiz - Professional Analytics',
    autoHideMenuBar: false
  });
  
  // Ensure icon is set properly on all platforms
  mainWindow.setIcon(iconPath);
  
  // Show Electron menu on all platforms
  mainWindow.setMenuBarVisibility(true);

  // Load the app
  mainWindow.loadFile('index.html');

  // Show window when ready and close splash screen
  mainWindow.once('ready-to-show', () => {
    // Minimum display time for splash screen (8 seconds)
    const minDisplayTime = 8000;
    
    const checkAndClose = () => {
      if (splashStartTime) {
        const elapsed = Date.now() - splashStartTime;
        const remainingTime = Math.max(0, minDisplayTime - elapsed);
        
        setTimeout(() => {
          if (splashWindow) {
            // Trigger fade-out animation in splash screen
            splashWindow.webContents.executeJavaScript(`
              document.getElementById('splashBox').classList.add('fade-out');
            `);
            
            // Close splash after fade animation completes
            setTimeout(() => {
              if (splashWindow) {
                splashWindow.close();
                splashWindow = null;
              }
              // Fade-in main window
              mainWindow.setOpacity(0);
              mainWindow.show();
              mainWindow.focus();
              
              // Fade in effect
              let opacity = 0;
              const fadeInterval = setInterval(() => {
                opacity += 0.1;
                if (opacity >= 1) {
                  mainWindow.setOpacity(1);
                  clearInterval(fadeInterval);
                } else {
                  mainWindow.setOpacity(opacity);
                }
              }, 30);
            }, 500); // Wait for fade-out animation (0.5s)
          } else {
            mainWindow.show();
            mainWindow.focus();
          }
        }, remainingTime);
      } else {
        // If splash hasn't shown yet, wait a bit and check again
        setTimeout(checkAndClose, 100);
      }
    };
    
    checkAndClose();
  });

  // Open DevTools in development
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }
}

// Function to start the Python backend
function startBackend() {
  // Get the app path - works for both development, packaged, and installed apps
  let appPath;
  let backendPath;
  
  // Try multiple possible locations for the backend
  const possiblePaths = [];
  
  if (app.isPackaged) {
    // For electron-builder packaged app
    const unpackedPath = path.join(process.resourcesPath, 'app.asar.unpacked', 'backend');
    const resourcesPath = path.join(process.resourcesPath, 'backend');
    possiblePaths.push(unpackedPath);
    possiblePaths.push(resourcesPath);
  }
  
  // For installed app, backend might be next to the executable
  const exeDirBackend = path.join(path.dirname(process.execPath), 'backend');
  possiblePaths.push(exeDirBackend);
  
  // For development
  const devBackend = path.join(__dirname, 'backend');
  possiblePaths.push(devBackend);
  
  // Find the first path that exists
  backendPath = possiblePaths.find(p => fs.existsSync(p));
  
  if (!backendPath) {
    console.error('Backend not found in any of these locations:', possiblePaths);
    dialog.showErrorBox(
      'Backend Error',
      `Python backend folder not found.\n\n` +
      `Searched in:\n${possiblePaths.join('\n')}\n\n` +
      `Please ensure the backend folder is properly installed.`
    );
    return;
  }
  
  appPath = path.dirname(backendPath);
  if (isDevelopment) {
    console.log('Found backend at:', backendPath);
  }
  
  const runScript = path.join(backendPath, 'run.py');
  const appScript = path.join(backendPath, 'app.py');
  
  // Check which script exists
  let scriptToRun = fs.existsSync(runScript) ? runScript : appScript;
  
  if (!fs.existsSync(scriptToRun)) {
    console.error('Backend script not found:', scriptToRun);
    dialog.showErrorBox(
      'Backend Error',
      `Python backend script not found at:\n${scriptToRun}\n\nPlease ensure the backend folder is properly installed.`
    );
    return;
  }

  // Determine Python command - try multiple options for Windows
  // Priority: py (Python Launcher) -> python -> python3 -> python.exe
  let pythonCmd;
  const pythonCommands = process.platform === 'win32' 
    ? ['py', 'python', 'python3', 'python.exe']
    : ['python3', 'python'];
  
  pythonCmd = pythonCommands[0]; // Start with first option
  
  // Try to find Python in common locations or PATH
  const requirementsPath = path.join(appPath, 'requirements.txt');
  if (fs.existsSync(requirementsPath)) {
    if (isDevelopment) {
      console.log('Found requirements.txt, backend should work');
    }
  }
  
  if (isDevelopment) {
    console.log('Starting Python backend:', scriptToRun);
    console.log('Backend path:', backendPath);
    console.log('Python command:', pythonCmd);
    console.log('Trying Python commands in order:', pythonCommands.join(', '));
  }
  
  // Start the backend process
  const spawnOptions = {
    cwd: backendPath,
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: true,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      FLASK_DEBUG: isDevelopment ? 'true' : 'false'  // Enable Flask debug mode in development
    }
  };
  
  // Quote the script path to handle spaces on Windows
  const quotedScriptPath = process.platform === 'win32' 
    ? `"${scriptToRun}"` 
    : scriptToRun;
  
  // Try the first Python command
  backendProcess = spawn(pythonCmd, [quotedScriptPath], spawnOptions);

  // Track backend process state
  let backendStarted = false;
  let backendReady = false;
  let lastOutputTime = Date.now();

  // Monitor process health - always log health status
  const healthCheckInterval = setInterval(() => {
    if (backendProcess && !backendProcess.killed) {
      const timeSinceLastOutput = Date.now() - lastOutputTime;
      if (timeSinceLastOutput > 30000) {
        // Log if no output for 30 seconds (but process still alive)
        console.log(`[Backend Health] Process alive, last output ${Math.floor(timeSinceLastOutput / 1000)}s ago`);
      }
    } else {
      clearInterval(healthCheckInterval);
    }
  }, 10000); // Check every 10 seconds

  // Log ALL backend output - show everything for debugging
  backendProcess.stdout.on('data', (data) => {
    const output = data.toString();
    lastOutputTime = Date.now();
    
    // ALWAYS log all backend output - no filtering
    console.log(`Backend: ${output}`);
    
    // Check if backend started successfully
    if ((output.includes('Running on') || output.includes('Starting DataWhiz Backend') || output.includes('localhost:5000')) && !backendStarted) {
      backendStarted = true;
      console.log('✓ Backend process started successfully');
    }
    
    // Check if backend is ready (Flask debugger active or server running)
    if ((output.includes('Debugger is active') || output.includes('Running on http://')) && !backendReady) {
      backendReady = true;
      console.log('✓ Backend is ready and responding');
    }
  });

  backendProcess.stderr.on('data', (data) => {
    const error = data.toString();
    lastOutputTime = Date.now();
    
    // ALWAYS log all backend errors and warnings - show everything
    console.error(`Backend Error: ${error}`);
    
    // Don't show dialog for warnings or info messages (but still log them)
    if (error.includes('WARNING') || error.includes('INFO') || error.includes('[INFO]') || error.includes('[WARNING]')) {
      return;
    }
  });

  backendProcess.on('error', (error) => {
    console.error('Failed to start backend with command:', pythonCmd);
    console.error('Error:', error.message);
    
    // If it's a "command not found" error and we have other commands to try
    if (error.code === 'ENOENT' && pythonCommands.length > 1) {
      const nextCmd = pythonCommands[1];
      pythonCommands.shift(); // Remove failed command
      if (isDevelopment) {
        console.log(`Trying next Python command: ${nextCmd}`);
      }
      
      // Retry with next command
      try {
        // Clear previous health check if exists
        if (healthCheckInterval) {
          clearInterval(healthCheckInterval);
        }
        
        backendProcess = spawn(nextCmd, [quotedScriptPath], spawnOptions);
        
        // Reset state for retry
        backendStarted = false;
        backendReady = false;
        lastOutputTime = Date.now();
        
        // Re-attach improved handlers with same logging
        const retryHealthCheck = setInterval(() => {
          if (backendProcess && !backendProcess.killed) {
            const timeSinceLastOutput = Date.now() - lastOutputTime;
            if (timeSinceLastOutput > 30000) {
              console.log(`[Backend Health] Process alive, last output ${Math.floor(timeSinceLastOutput / 1000)}s ago`);
            }
          } else {
            clearInterval(retryHealthCheck);
          }
        }, 10000);
        
        backendProcess.stdout.on('data', (data) => {
          const output = data.toString();
          lastOutputTime = Date.now();
          
          // ALWAYS log all backend output
          console.log(`Backend: ${output}`);
          
          if ((output.includes('Running on') || output.includes('Starting DataWhiz Backend') || output.includes('localhost:5000')) && !backendStarted) {
            backendStarted = true;
            console.log('✓ Backend process started successfully');
          }
          
          if ((output.includes('Debugger is active') || output.includes('Running on http://')) && !backendReady) {
            backendReady = true;
            console.log('✓ Backend is ready and responding');
          }
        });
        
        backendProcess.stderr.on('data', (data) => {
          const error = data.toString();
          lastOutputTime = Date.now();
          
          // ALWAYS log all backend errors
          console.error(`Backend Error: ${error}`);
          
          if (error.includes('WARNING') || error.includes('INFO')) {
            return;
          }
        });
        
        backendProcess.on('spawn', () => {
          console.log(`[Backend] Retry: Process spawned with PID: ${backendProcess.pid}`);
          console.log(`[Backend] Retry: Command: ${nextCmd} ${quotedScriptPath}`);
        });
        
        backendProcess.on('error', (retryError) => {
          clearInterval(retryHealthCheck);
          console.error('[Backend] All Python commands failed');
          showPythonError(retryError, nextCmd);
        });
        
        backendProcess.on('exit', (code, signal) => {
          clearInterval(retryHealthCheck);
          console.log(`[Backend] Process exited with code: ${code}, signal: ${signal}`);
          if (code !== null && code !== 0 && code !== 143 && code !== 1) {
            console.error(`[Backend] Process exited unexpectedly with code ${code}`);
            if (!appIsQuiting) {
              dialog.showErrorBox(
                'Backend Stopped',
                `The Python backend stopped unexpectedly (exit code: ${code}).\n\nThe application may not function correctly.`
              );
            }
          }
          backendProcess = null;
          backendStarted = false;
          backendReady = false;
        });
        
        return; // Don't show error dialog, retrying
      } catch (retryErr) {
        showPythonError(error, pythonCmd);
      }
    } else {
      showPythonError(error, pythonCmd);
    }
  });
  
  function showPythonError(error, cmd) {
    dialog.showErrorBox(
      'Backend Error',
      `Failed to start Python backend with command: ${cmd}\n\n` +
      `Error: ${error.message}\n\n` +
      `Please ensure:\n` +
      `1. Python 3.11+ is installed\n` +
      `2. Python is added to your PATH\n` +
      `3. Try running: py --version or python --version in Command Prompt\n` +
      `4. Required packages are installed: pip install -r requirements.txt`
    );
  }

  backendProcess.on('exit', (code, signal) => {
    clearInterval(healthCheckInterval);
    
    // Always log exit details
    console.log(`[Backend] Process exited with code: ${code}, signal: ${signal}`);
    
    if (code !== null && code !== 0 && code !== 143 && code !== 1) {
      console.error(`[Backend] Process exited unexpectedly with code ${code}`);
      // Only show error if it wasn't a clean shutdown
      if (!appIsQuiting) {
        dialog.showErrorBox(
          'Backend Stopped',
          `The Python backend stopped unexpectedly (exit code: ${code}).\n\nThe application may not function correctly.`
        );
      }
    } else if (code === 0 || code === 143) {
      console.log('[Backend] Process stopped cleanly');
    }
    backendProcess = null;
    backendStarted = false;
    backendReady = false;
  });
  
  // Monitor process spawn - always log process details
  backendProcess.on('spawn', () => {
    console.log(`[Backend] Process spawned with PID: ${backendProcess.pid}`);
    console.log(`[Backend] Command: ${pythonCmd} ${quotedScriptPath}`);
    console.log(`[Backend] Working directory: ${backendPath}`);
    console.log(`[Backend] Process environment: FLASK_DEBUG=${spawnOptions.env.FLASK_DEBUG}`);
  });
}

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
  // Set app icon and remove Electron branding
  const iconPath = path.join(__dirname, 'mind-map.png');
  
  // Set the app user model ID for Windows (removes default Electron icon)
  if (process.platform === 'win32') {
    app.setAppUserModelId('com.datawhiz.app');
  }
  
  // Set the dock icon for macOS
  if (process.platform === 'darwin') {
    app.dock.setIcon(iconPath);
  }
  
  // Start the Python backend first
  startBackend();
  
  // Wait a moment for backend to start
  setTimeout(() => {
    // Create custom menu without Developer Tools
    createMenu();
    
    // Handle save dialog requests
    ipcMain.handle('show-save-dialog', async (event, options) => {
      const result = await dialog.showSaveDialog(mainWindow, options);
      return result;
    });
    
    // Show splash screen first
    createSplashWindow();
    
    // Then create main window (will load in background)
    createWindow();
  }, 2000); // Wait 2 seconds for backend to initialize

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      splashStartTime = null; // Reset for new splash window
      createSplashWindow();
      createWindow();
    }
  });
});

// Quit when all windows are closed
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Cleanup: Kill backend process when app quits
app.on('before-quit', () => {
  appIsQuiting = true;
  if (backendProcess) {
    console.log('Stopping Python backend...');
    if (process.platform === 'win32') {
      // On Windows, kill the entire process tree
      try {
        spawn('taskkill', ['/F', '/T', '/PID', backendProcess.pid.toString()], {
          stdio: 'ignore',
          shell: true
        });
      } catch (error) {
        console.error('Error killing backend process:', error);
      }
    } else {
      // On Unix-like systems, use kill
      try {
        backendProcess.kill('SIGTERM');
        // If SIGTERM doesn't work, wait a bit then force kill
        setTimeout(() => {
          if (backendProcess) {
            backendProcess.kill('SIGKILL');
          }
        }, 2000);
      } catch (error) {
        console.error('Error killing backend process:', error);
      }
    }
    backendProcess = null;
  }
});