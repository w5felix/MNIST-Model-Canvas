"""DigitsVis package

A modular MNIST training and web inference demo.

Submodules:
- digitsvis.models: Model architectures (SimpleCNN)
- digitsvis.data: Dataset registry and dataloader helpers
- digitsvis.inference: Preprocessing and model loading utilities
- digitsvis.web: Flask app factory for serving the demo
"""
from .models import SimpleCNN  # re-export for convenience
from .inference import preprocess_image, load_model  # re-export
from .web import create_app  # re-export

__all__ = [
    "SimpleCNN",
    "preprocess_image",
    "load_model",
    "create_app",
]
