import os
from typing import List

import cv2
import numpy as np
from PIL import Image


def imread(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def imwrite(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    else:
        encode_param = []
    success, buf = cv2.imencode(ext if ext else ".png", img, encode_param)
    if not success:
        raise ValueError(f"Failed to encode image for saving: {path}")
    with open(path, 'wb') as f:
        buf.tofile(f)


def to_qimage_bgr(img: np.ndarray):
    """Convert BGR numpy image to QImage (PyQt5)."""
    from PyQt5.QtGui import QImage  # local import to avoid GUI dep in non-GUI flows
    if img is None:
        return None
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        return qimg.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return qimg.copy()


def list_images_in_folder(folder: str) -> List[str]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths: List[str] = []
    for name in os.listdir(folder):
        if os.path.splitext(name)[1].lower() in exts:
            paths.append(os.path.join(folder, name))
    paths.sort()
    return paths
