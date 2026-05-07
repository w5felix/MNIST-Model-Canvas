DigitsVis

Purpose
- Train a simple CNN on the MNIST dataset and demo real-time digit recognition in the browser.

Main parts
- Training: train.py (PyTorch)
- Web app: app.py + static/ (Flask backend + canvas UI)
- Core modules (package): digitsvis/ (models, data registry, inference utils, app factory)

Quick start
1) Install
- Create and activate a virtual environment, then install requirements:
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
  pip install -r requirements.txt

2) Train (produces mnist_cnn.pt at project root, and also exports ONNX to static/mnist_cnn.onnx)
  python train.py
  # Options:
  #   --skip-onnx           Skip exporting ONNX after training
  #   --onnx-out <path>     Custom path for the ONNX file (default: static/mnist_cnn.onnx)
  #   --export-only         Do not train; export ONNX from an existing checkpoint at --model-out

3) Run the web app (Python server)
  python app.py
  # Open http://localhost:5777
  # Health check: http://localhost:5777/health

How inference works
- In-browser (standalone): If static/mnist_cnn.onnx exists, the page loads it with onnxruntime-web and runs fully in the browser.
- Server fallback: If the ONNX file is missing, the page automatically falls back to POST /predict on the Python server.

Configure
- Environment variables (optional):
  HOST=0.0.0.0 PORT=5777 FLASK_DEBUG=1  # server options
  MODEL_PATH=mnist_cnn.pt               # model checkpoint path
  STATIC_FOLDER=static                  # static files folder

Dataset hooking (modular)
- The training pipeline uses a dataset registry (digitsvis.data) so you can plug in a different dataset without editing training code.
  Option A: Use built-in MNIST (default):
    python train.py --dataset mnist --data-root data

  Option B: Register your own dataset in code:
    # my_dataset.py
    from digitsvis.data import register_dataset
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    @register_dataset("mydigits")
    def build(root: str = "data", batch_size: int = 64, num_workers: int = 2):
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)  # replace with yours
        test = datasets.MNIST(root=root, train=False, download=True, transform=tfm)  # replace with yours
        return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers))

    # Train with your dataset by importing the plugin first:
    python train.py --dataset mydigits --dataset-plugin my_dataset

Notes
- The minimal frontend is served from static/. No separate build step required.
- The Flask app lazily loads the model on the first /predict call.



Deploying on Render
- This repo includes a Render blueprint (render.yaml). Push to your own Git repo and create a Blueprint on Render using that URL.
- See RENDER_DEPLOY.md for step-by-step instructions and tips.
