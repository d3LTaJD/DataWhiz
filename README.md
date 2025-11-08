# DataWhiz - Professional Data Analytics Platform

A modern desktop application for data science and analytics built with Electron and Python.

## Features

- 📊 **Data Management** - Import CSV, Excel, JSON files
- 📈 **Statistical Analysis** - Advanced statistical methods
- 📊 **Visualizations** - Interactive charts with Plotly
- 🤖 **Machine Learning** - Classification and regression models
- 🎨 **Modern UI** - Professional desktop interface

## Quick Start

### Option 1: Simple Start
```bash
start.bat
```

### Option 2: Manual Start
1. Start Python backend:
   ```bash
   py backend/app.py
   ```
2. Start Electron frontend:
   ```bash
   npx electron .
   ```

## Installation

1. Install Node.js from https://nodejs.org/
2. Install Python dependencies:
   ```bash
   py -m pip install flask flask-cors pandas numpy scikit-learn plotly openpyxl
   ```
3. Install Electron dependencies:
   ```bash
   npm install
   ```

## Project Structure

```
datawhiz/
├── backend/           # Python Flask API
│   └── app.py        # Backend server
├── assets/           # Application assets
├── uploads/          # User uploaded files
├── index.html        # Main application UI
├── styles.css        # Modern styling
├── script.js         # Frontend logic
├── main.js           # Electron main process
├── package.json      # Node.js dependencies
└── start.bat         # Quick launcher
```

## Technology Stack

- **Frontend**: Electron + HTML/CSS/JavaScript
- **Backend**: Python Flask API
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn

## License

MIT License - Built for professional data analytics.