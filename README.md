# DataWhiz – Professional Data Analytics Platform

DataWhiz is a cross-platform desktop analytics studio that combines a Python Flask backend with an Electron-powered user interface. It streamlines data ingestion, cleaning, visualization, and machine-learning experimentation for business, finance, and healthcare datasets.

## Highlights

- 📂 **Multi-format ingestion** – Upload CSV/Excel datasets and persist cleaned outputs for rapid iteration.
- 🧮 **End-to-end analytics** – Execute preprocessing, statistical summaries, forecasting, and ML pipelines directly from the app.
- 📊 **Interactive dashboards** – Explore Plotly-driven charts spanning business, finance, and healthcare use cases.
- ⚙️ **Modular architecture** – Python services handle heavy lifting while Electron orchestrates a polished UX with reusable dashboards.
- 🚀 **One-click launchers** – Bundled scripts start both backend and frontend, no manual orchestration required.

## Architecture Overview

- **Electron shell (`main.js`)** boots the desktop app, loads the web assets, and communicates with the backend.
- **Frontend assets** (`index.html`, `script.js`, `styles/theme.css`, and topic-specific dashboards) render analysis workflows and user-facing dashboards.
- **Flask backend (`backend/app`)** exposes REST endpoints for uploading datasets, computing analytics, and serving transformed data.
- **Data processing core (`backend/app/core/data_processing.py`)** centralizes cleaning, feature engineering, ML model execution, and serialization.
- **Dataset samples (`datasets/`)** provide ready-to-explore CSVs and metadata to test-drive the experience.

```text
datawhiz/
├── backend/
│   ├── app.py                 # Backend entry point
│   └── app/
│       ├── api/               # Flask Blueprints (analytics, business, healthcare, upload)
│       └── core/              # Data transformation & ML helpers
├── assets/                    # Icons and branding
├── datasets/                  # Sample data and catalog definitions
├── frontend html dashboards   # Vertical-specific views (business, finance, healthcare)
├── scripts & launchers        # start.bat, start.ps1, install_electron.bat, etc.
├── main.js                    # Electron main process
├── package.json               # Node/Electron dependencies
├── requirements.txt           # Python backend dependencies
└── README.md
```

## Prerequisites

- Node.js 18+ (for Electron and frontend tooling)
- Python 3.11+ (for Flask backend)
- Git LFS is *not* required – build artifacts are excluded from the repository.

## Setup

1. Install Python dependencies (virtual environment recommended):
   ```bash
   py -m pip install --upgrade pip
   py -m pip install -r requirements.txt
   ```

2. Install Node/Electron dependencies:
   ```bash
   npm install
   ```

3. (Optional) Configure environment variables:
   - `FLASK_ENV=development` for debug logging
   - `PORT` to change the backend port (default 5000)

## Running the App

### Fast path (Windows)

```powershell
start.bat
```
This script spins up the Flask backend and launches the Electron shell with sensible defaults.

### Manual workflow

1. Start the backend API:
   ```bash
   py backend/run.py
   ```

2. Launch the Electron app (in a second terminal):
   ```bash
   npm start
   ```
   or, for a production-style build:
   ```bash
   npx electron .
   ```

## Working with Data

- Upload raw files through the UI or drop them into `backend/uploads/` for quick access.
- Processed outputs (cleaned, merged datasets) are persisted alongside the originals, enabling reproducible analytics flows.
- Sample dashboards under `business-analysis/`, `finance-analysis/`, and `healthcare-analysis/` demonstrate sector-specific insights you can tailor to your own data.

## Development Tips

- **Python linting**: integrate `flake8`/`black` for consistent backend formatting.
- **Frontend hot reload**: use `npm run dev` if you add a Vite/Webpack flow (currently static assets served via Electron).
- **Packaging**: build bundles with your preferred Electron builder; ensure binaries land in `dist/` (already ignored by git).

## Roadmap Ideas

- Add authentication for multi-user environments.
- Expand ML templates (time-series forecasting, anomaly detection).
- Integrate cloud storage connectors for dataset ingestion.

## License

MIT License – see the license header for attribution requirements. Feel free to adapt DataWhiz for internal analytics teams or client-facing solutions.