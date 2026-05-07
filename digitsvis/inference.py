import io
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from .models import SimpleCNN


def load_model(model_path: str = "mnist_cnn.pt", device: Optional[torch.device] = None) -> nn.Module:
    """Load a SimpleCNN from a checkpoint path.

    Accepts either:
    - a dict with key 'model_state_dict'
    - a raw state_dict mapping
    - or a full nn.Module (will be used as-is)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)

    # If a full model was saved, prefer using it directly
    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
        model.eval()
        return model

    # Otherwise, construct our architecture and load weights
    model = SimpleCNN().to(device)
    state = ckpt.get('model_state_dict', None) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise TypeError("Unsupported checkpoint format: expected dict with 'model_state_dict' or a state_dict mapping or an nn.Module")
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Preprocess a user-drawn digit according to the 8-step pipeline:
    1) Grayscale, 2) Inversion check (ensure white digit on black), 3) BBox,
    4) Crop, 5) Resize to fit within ~20x20, 6) Pad to 28x28, 7) Center by COM,
    8) Normalize to [0,1]. Returns a torch tensor of shape [1,1,28,28].
    """
    # 1) Convert to grayscale (L)
    img = pil_img.convert('L')

    # Work in numpy (0..255)
    arr = np.asarray(img).astype(np.float32)

    # 2) Inversion check: ensure white digit on black background.
    # Heuristic: if more than half the pixels are bright, assume white background -> invert.
    bright_fraction = float((arr > 127).mean()) if arr.size else 0.0
    if bright_fraction > 0.5:
        arr = 255.0 - arr

    # 3) Find bounding box of the digit using a small threshold
    th = 20.0  # tolerance for non-background
    ys, xs = np.where(arr > th)
    if ys.size == 0 or xs.size == 0:
        # Empty drawing: return zeros 28x28
        arr28 = np.zeros((28, 28), dtype=np.float32)
        tensor = torch.from_numpy(arr28)[None, None, :, :]
        return tensor

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # 4) Crop to bounding box
    arr_crop = arr[y0:y1 + 1, x0:x1 + 1]

    # 5) Resize so the longest side fits about 20 pixels
    h, w = arr_crop.shape
    if max(h, w) == 0:
        arr28 = np.zeros((28, 28), dtype=np.float32)
        tensor = torch.from_numpy(arr28)[None, None, :, :]
        return tensor
    scale = 20.0 / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img_small = Image.fromarray(arr_crop.astype(np.uint8), mode='L').resize((new_w, new_h), Image.BILINEAR)
    small = np.asarray(img_small).astype(np.float32)

    # 6) Pad to 28x28 (centered)
    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small

    # 7) Center by center of mass (integer pixel shift)
    mass = canvas.sum()
    if mass > 1e-6:
        ys_grid, xs_grid = np.mgrid[0:28, 0:28]
        cy = float((canvas * ys_grid).sum() / mass)
        cx = float((canvas * xs_grid).sum() / mass)
        # Aim for center at 13.5 (center between 13 and 14)
        dy = int(round(13.5 - cy))
        dx = int(round(13.5 - cx))
        if dx != 0 or dy != 0:
            shifted = np.zeros_like(canvas)
            H, W = canvas.shape
            src_x0 = max(0, -dx)
            src_y0 = max(0, -dy)
            dst_x0 = max(0, dx)
            dst_y0 = max(0, dy)
            w_copy = min(W - src_x0, W - dst_x0)
            h_copy = min(H - src_y0, H - dst_y0)
            if w_copy > 0 and h_copy > 0:
                shifted[dst_y0:dst_y0 + h_copy, dst_x0:dst_x0 + w_copy] = \
                    canvas[src_y0:src_y0 + h_copy, src_x0:src_x0 + w_copy]
            canvas = shifted

    # 8) Normalize to [0,1]
    arr28 = (canvas / 255.0).astype(np.float32)

    # To 1x1x28x28 tensor
    tensor = torch.from_numpy(arr28)[None, None, :, :]
    return tensor
