from typing import Literal
import cv2
import numpy as np


def denoise(
    img_bgr: np.ndarray,
    method: Literal["fastnlm", "bilateral", "nlm"] = "fastnlm",
    strength: float = 0.5,
) -> np.ndarray:
    """
    Denoise an image using different methods.
    strength in [0,1], mapped to algorithm-specific params.
    """
    strength = float(np.clip(strength, 0.0, 1.0))

    if method == "fastnlm":
        # OpenCV fastNlMeansDenoisingColored
        h = 3 + int(12 * strength)  # 3..15
        template = 7
        search = 21
        return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, template, search)

    elif method == "bilateral":
        d = 5 + int(10 * strength)  # 5..15
        sigma_color = 25 + int(125 * strength)  # 25..150
        sigma_space = 25 + int(125 * strength)
        return cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    elif method == "nlm":
        # Non-local means via OpenCV photo module (grayscale)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h = 5 + int(15 * strength)
        den = cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)
        return cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)

    else:
        raise ValueError(f"Unknown denoise method: {method}")
