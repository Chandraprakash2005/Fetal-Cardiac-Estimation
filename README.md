# Fetal-Cardiac-Estimation 🫀

An advanced AI-powered Fetal ECG analysis and Cardiac Age Estimation platform integrating **Multivariate Empirical Mode Decomposition (MEMD)**, **Physics-Informed Neural Networks (PINN)**, and a **Multi-Scale Temporal Transformer**.

---

## 🚀 Key Features

- **Non-Invasive Fetal ECG Extraction**: Eliminates maternal ECG (mECG) baseline artifacts using MEMD.
- **PINN Electrodynamic Conservation**: Reconstructs subtle fetal QRS complexes constrained by cardiac physical laws.
- **Multi-Scale Feature Extraction**: Extracts high-dimensional morphologic biomarkers and R-R interval agreement metrics.
- **Deep Learning Cardiac Age Estimator**: Estimates fetal cardiac maturation age with high confidence intervals.
- **Sub-Second Streaming Pipeline**: Interactive dashboard with real-time SSE progress, telemetry feedback, and responsive tactile UI.

---

## 🛠️ Architecture

```
Fetal-Cardiac-Estimation/
├── Backend/              # FastAPI server, signal preprocessing, MEMD + PINN inference
│   ├── main.py           # REST + SSE Streaming endpoints
│   ├── src/              # Pipeline modules (preprocessing, models, evaluation)
│   └── requirements.txt  # Python dependencies
├── frontend/             # Vite + React tactile neumorphic dashboard
│   ├── src/              # React components, pipeline visualizer, telemetry
│   └── package.json      # Node.js dependencies
└── sample_ecg/           # Benchmark ECG datasets (.dat, .hea, .edf)
```

---

## ⚡ Quick Start

### 1. Backend Setup
```bash
cd Backend
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173` to launch the analyzer.
