from typing import Optional
import threading
import os

import cv2
import numpy as np
from PIL import Image

# Lazy Real-ESRGAN loader to avoid heavy imports if unused
__model = None
__lock = threading.Lock()


def _load_model(scale: int = 2):
    global __model
    with __lock:
        if __model is not None and getattr(__model, "scale", None) == scale:
            return __model
        # import inside to avoid hard dependency unless SR is used
        from py_real_esrgan.model import RealESRGAN
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        __model = RealESRGAN(device, scale=scale)
        # ensure weights directory exists; the API downloads when missing
        weights_path = f"weights/RealESRGAN_x{scale}.pth"
        __model.load_weights(weights_path)
        __model.scale = scale
        return __model


def super_resolve(img_bgr: np.ndarray, scale: int = 2) -> Optional[np.ndarray]:
    # Allow 2x, 4x, 8x. Fallback to closest supported value
    req = int(scale)
    if req >= 8:
        scale = 8
    elif req >= 4:
        scale = 4
    else:
        scale = 2
    model = _load_model(scale)

    # Ensure RGB PIL input
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_in = Image.fromarray(img_rgb)

    sr_out = model.predict(pil_in)
    # Convert output to numpy RGB regardless of return type
    if isinstance(sr_out, Image.Image):
        sr_rgb = np.array(sr_out)
    else:
        # assume numpy RGB
        sr_rgb = sr_out

    if sr_rgb.dtype != np.uint8:
        sr_rgb = np.clip(sr_rgb, 0, 255).astype(np.uint8)

    sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
    return sr_bgr
