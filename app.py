import os
from digitsvis import create_app

# Create the Flask app using the factory; configurable via env vars
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolve static folder and model path relative to this file to avoid CWD issues
STATIC_FOLDER = os.getenv("STATIC_FOLDER", "static")
if not os.path.isabs(STATIC_FOLDER):
    STATIC_FOLDER = os.path.join(BASE_DIR, STATIC_FOLDER)

MODEL_PATH = os.getenv("MODEL_PATH", "mnist_cnn.pt")
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)

app = create_app(static_folder=STATIC_FOLDER, model_path=MODEL_PATH)

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5777'))
    debug = os.getenv('FLASK_DEBUG', '1') == '1'
    app.run(host=host, port=port, debug=debug)
