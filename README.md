# Photo Restoration Desktop App (Denoise, Deblur, Super-Resolution)

## Features
- Load a single image or a folder of images (JPG/PNG).
- Processing modes:
  - Denoise (FastNlMeans / Bilateral / Non-Local Means)
  - Deblur (Wiener deconvolution or Unsharp mask)
  - Super-Resolution (Real-ESRGAN, 2x/4x)
- Adjustable parameters: strength level, upscale factor.
- Side-by-side preview (original vs processed) and save results.

## Setup
1. Create a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Torch can be heavy. If you don't need SR, you may skip torch/Real-ESRGAN. For CPU-only Torch on Windows:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Run
```bash
python app.py
```

## Notes
- Super-resolution model will be loaded lazily on first use (it may take time to download weights).
- Supported images: JPG, JPEG, PNG.
- Outputs are saved to `output/` by default. You can change it in the UI when saving.

## Project Structure
- `app.py` — PyQt5 GUI
- `processing/` — image processing functions
  - `denoise.py`
  - `deblur.py`
  - `sr.py`
- `utils/`
  - `image_io.py`
