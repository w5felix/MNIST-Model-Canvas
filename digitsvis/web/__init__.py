from pathlib import Path
import base64
import io

from flask import Flask, jsonify, request
from PIL import Image
import torch

from ..inference import load_model, preprocess_image


def create_app(static_folder: str = 'static', model_path: str = 'mnist_cnn.pt') -> Flask:
    # Resolve folders to absolute paths to avoid CWD/package-root ambiguities
    static_path = Path(static_folder).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()

    app = Flask(__name__, static_folder=str(static_path), static_url_path="/static")

    MODEL_PATH = model_path
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @app.route('/')
    def index():
        # Serve the main page from the configured static folder
        return app.send_static_file('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        nonlocal model
        if model is None:
            if not MODEL_PATH.exists():
                return jsonify({"error": "Model not found. Please run train.py first to create mnist_cnn.pt"}), 400
            model = load_model(str(MODEL_PATH), device=device)

        data = request.get_json(silent=True) or {}
        data_url = data.get('image')
        if not data_url or not isinstance(data_url, str) or 'base64,' not in data_url:
            return jsonify({"error": "Invalid image payload"}), 400

        try:
            b64 = data_url.split('base64,', 1)[1]
            img_bytes = base64.b64decode(b64)
            pil_img = Image.open(io.BytesIO(img_bytes))
            tensor = preprocess_image(pil_img).to(device)
            with torch.no_grad():
                logits = model(tensor)
                pred = int(logits.argmax(dim=1).item())
                probs = torch.softmax(logits, dim=1).cpu().numpy().tolist()[0]
            return jsonify({"prediction": pred, "probs": probs})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/health')
    def health():
        onnx_path = static_path / 'mnist_cnn.onnx'
        return jsonify({
            "status": "ok",
            "device": str(device),
            "model_exists": MODEL_PATH.exists(),
            "model_path": str(MODEL_PATH),
            "onnx_exists": onnx_path.exists(),
            "onnx_path": str(onnx_path),
        })

    # Rely on Flask's built-in static file handling at /static/*

    return app
