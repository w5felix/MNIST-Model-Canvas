Render deployment guide

Overview
- This repository includes a Render blueprint (render.yaml) and is ready to deploy as a Python Web Service.
- The web UI runs fully in the browser using ONNX Runtime Web when static/mnist_cnn.onnx is present. The server only needs to serve static files and a health endpoint.
- A server-side /predict fallback exists and uses PyTorch; it's only used if the ONNX file is missing or fails to load.

What you need
- A Git repository hosting this project (GitHub/GitLab/Bitbucket).
- A Render account (free plan is fine for this demo).

Files used by Render
- render.yaml: Defines a web service with build and start commands and health check.
- app.py: Exposes a Flask app object (app) consumed by gunicorn.
- static/: Frontend assets. static/mnist_cnn.onnx enables in-browser inference.

Quick deploy via Blueprint (recommended)
1) Push this project to your own Git repo.
2) Ensure static/mnist_cnn.onnx is present in the repo. If not, run locally: 
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python train.py  # exports static/mnist_cnn.onnx by default
   git add static/mnist_cnn.onnx mnist_cnn.pt && git commit -m "Add model artifacts"
3) In Render, click New > Blueprint and paste your repository URL. Render will read render.yaml.
4) Accept defaults or adjust:
   - Name: digitsvis (or any)
   - Region: Oregon (default in render.yaml)
   - Plan: Free
   - Health check path: /health
5) Click Apply. Build will:
   - pip install -r requirements.txt
   - Start: gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT app:app
6) Once live, open the service URL. You should see the drawing UI. The status in the header should read: "ONNX Runtime (WebAssembly) ready".

Manual deploy (without blueprint)
1) New > Web Service > Build and deploy from a Git repository.
2) Environment: Python
3) Build command: pip install -r requirements.txt
4) Start command: gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT app:app
5) Health check path: /health
6) Optional environment variables:
   - STATIC_FOLDER=static
   - MODEL_PATH=mnist_cnn.pt
   - FLASK_DEBUG=0

Notes and tips
- Using in-browser ONNX avoids server-side PyTorch inference; the server can run on very small instances. Keep static/mnist_cnn.onnx in the repo for best performance.
- If you prefer server-side inference, make sure mnist_cnn.pt is present (default path) or set MODEL_PATH accordingly. The first /predict call will lazily load the model.
- Data files under data/MNIST/ are not required at runtime and can be removed from the repo to shrink it; they are only needed for training.
- You can change the Python version via env var PYTHON_VERSION in render.yaml.
- Health endpoint: https://<your-service>.onrender.com/health

Local development
- Run locally with: 
  python app.py  # binds to 0.0.0.0:5777 by default; respects HOST and PORT env vars
- Train or re-export ONNX: 
  python train.py  # creates mnist_cnn.pt and static/mnist_cnn.onnx
