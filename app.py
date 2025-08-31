import os
from typing import List, Optional

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PyQt5 import QtCore, QtGui, QtWidgets

from utils.image_io import imread, imwrite, to_qimage_bgr, list_images_in_folder
from processing.denoise import denoise
from processing.deblur import deblur
from processing.sr import super_resolve


class ImageProcessor(QtCore.QObject):
    processed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

    @QtCore.pyqtSlot(object, str, float, str)
    def run(self, img: np.ndarray, mode: str, strength: float, extra: str):
        try:
            if mode == "Denoise":
                out = denoise(img, method=extra or "fastnlm", strength=strength)
            elif mode == "Deblur":
                out = deblur(img, method=extra or "wiener", strength=strength)
            elif mode == "Super-Resolution":
                if extra == "4x":
                    scale = 4
                else:
                    scale = 2
                out = super_resolve(img, scale=scale)
            else:
                raise ValueError("Unknown mode")
            self.processed.emit(out)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Restoration - Denoise/Deblur/SR")
        self.resize(1200, 700)

        self.image_paths: List[str] = []
        self.current_index: int = -1
        self.original: Optional[np.ndarray] = None
        self.processed: Optional[np.ndarray] = None

        self._build_ui()
        self._setup_worker()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Top controls
        controls = QtWidgets.QHBoxLayout()
        self.btn_open_img = QtWidgets.QPushButton("Open Image")
        self.btn_open_dir = QtWidgets.QPushButton("Open Folder")
        self.btn_prev = QtWidgets.QPushButton("◀ Prev")
        self.btn_next = QtWidgets.QPushButton("Next ▶")
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)

        controls.addWidget(self.btn_open_img)
        controls.addWidget(self.btn_open_dir)
        controls.addStretch(1)
        controls.addWidget(self.btn_prev)
        controls.addWidget(self.btn_next)
        layout.addLayout(controls)

        # Mode and params
        params = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Denoise", "Deblur", "Super-Resolution"])

        self.extra = QtWidgets.QComboBox()  # method or scale

        self.strength = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.strength.setRange(0, 100)
        self.strength.setValue(50)
        self.lbl_strength = QtWidgets.QLabel("Strength: 0.50")
        self.strength.valueChanged.connect(self._on_strength_change)
        # Initialize options after strength widgets exist
        self._update_extra_options("Denoise")
        self.mode.currentTextChanged.connect(self._update_extra_options)

        self.btn_process = QtWidgets.QPushButton("Process")
        self.btn_save = QtWidgets.QPushButton("Save Current")
        self.btn_save_all = QtWidgets.QPushButton("Save All (Folder)")
        self.lbl_metrics = QtWidgets.QLabel("PSNR: - | SSIM: -")
        self.lbl_metrics.setMinimumWidth(220)
        # Button styling
        btn_style = (
            "QPushButton { background:#2d2d2d; color:#eee; border:1px solid #444; padding:6px 12px; border-radius:4px }"
            "QPushButton:hover { background:#3a3a3a }"
            "QPushButton:disabled { color:#777; background:#222; border:1px solid #333 }"
        )
        for b in (self.btn_process, self.btn_save, self.btn_save_all, self.btn_open_img, self.btn_open_dir, self.btn_prev, self.btn_next):
            b.setStyleSheet(btn_style)

        params.addWidget(QtWidgets.QLabel("Mode:"))
        params.addWidget(self.mode)
        params.addWidget(QtWidgets.QLabel("Option:"))
        params.addWidget(self.extra)
        params.addWidget(self.lbl_strength)
        params.addWidget(self.strength)
        params.addStretch(1)
        params.addWidget(self.btn_process)
        params.addWidget(self.btn_save)
        params.addWidget(self.btn_save_all)
        layout.addLayout(params)

        # Image views as splitter for better resizing
        self.view_orig = QtWidgets.QLabel("Original")
        self.view_proc = QtWidgets.QLabel("Processed")
        for v in (self.view_orig, self.view_proc):
            v.setAlignment(QtCore.Qt.AlignCenter)
            v.setMinimumSize(400, 400)
            v.setStyleSheet("border:1px solid #333; background:#0f0f0f; color:#bbb")
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.view_orig)
        splitter.addWidget(self.view_proc)
        splitter.setSizes([1, 1])
        layout.addWidget(splitter, 1)

        # Signals
        self.btn_open_img.clicked.connect(self.open_image)
        self.btn_open_dir.clicked.connect(self.open_folder)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_process.clicked.connect(self.process_current)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_save_all.clicked.connect(self.save_all)

        # Status bar metrics
        self.statusBar().setStyleSheet("QStatusBar{background:#1b1b1b; color:#ddd}")
        self.statusBar().addPermanentWidget(self.lbl_metrics)

    def _setup_worker(self):
        self.thread = QtCore.QThread(self)
        self.worker = ImageProcessor()
        self.worker.moveToThread(self.thread)
        self.worker.processed.connect(self._on_processed)
        self.worker.failed.connect(self._on_failed)
        self.thread.start()

    def _update_extra_options(self, mode: str):
        self.extra.clear()
        if mode == "Denoise":
            self.extra.addItems(["fastnlm", "bilateral", "nlm"])
            self.strength.setEnabled(True)
            # refresh label to current value
            self._on_strength_change(self.strength.value())
        elif mode == "Deblur":
            self.extra.addItems(["wiener", "unsharp"])
            self.strength.setEnabled(True)
            self._on_strength_change(self.strength.value())
        else:
            self.extra.addItems(["2x", "4x"])
            self.strength.setEnabled(False)
            self.lbl_strength.setText("Strength: N/A")

    def _on_strength_change(self, val: int):
        self.lbl_strength.setText(f"Strength: {val/100:.2f}")

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self.image_paths = [path]
        self.current_index = 0
        self._load_current()
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Folder", "")
        if not folder:
            return
        paths = list_images_in_folder(folder)
        if not paths:
            QtWidgets.QMessageBox.warning(self, "No images", "No images found in this folder.")
            return
        self.image_paths = paths
        self.current_index = 0
        self._load_current()
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        n = len(self.image_paths)
        self.btn_prev.setEnabled(n > 1 and self.current_index > 0)
        self.btn_next.setEnabled(n > 1 and self.current_index < n - 1)

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._load_current()
            self._update_nav_buttons()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current()
            self._update_nav_buttons()

    def _load_current(self):
        path = self.image_paths[self.current_index]
        self.original = imread(path)
        self.processed = None
        self._show_image(self.view_orig, self.original)
        self.view_proc.setPixmap(QtGui.QPixmap())
        self.view_proc.setText("Processed")
        self.setWindowTitle(f"Photo Restoration - {os.path.basename(path)}")

    def _show_image(self, label: QtWidgets.QLabel, img: np.ndarray):
        qimg = to_qimage_bgr(img)
        pix = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(pix.scaled(label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.original is not None:
            self._show_image(self.view_orig, self.original)
        if self.processed is not None:
            self._show_image(self.view_proc, self.processed)

    def process_current(self):
        if self.original is None:
            return
        mode = self.mode.currentText()
        strength = self.strength.value() / 100.0
        extra = self.extra.currentText()
        # Invoke in the worker thread
        QtCore.QMetaObject.invokeMethod(
            self.worker,
            "run",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(object, self.original),
            QtCore.Q_ARG(str, mode),
            QtCore.Q_ARG(float, strength),
            QtCore.Q_ARG(str, extra),
        )
        self.btn_process.setEnabled(False)
        self.btn_process.setText("Processing...")

    def _on_processed(self, out: np.ndarray):
        self.processed = out
        self._show_image(self.view_proc, self.processed)
        # update metrics
        try:
            self._update_metrics()
        except Exception as _:
            self.lbl_metrics.setText("PSNR: - | SSIM: -")
        self.btn_process.setEnabled(True)
        self.btn_process.setText("Process")

    def _on_failed(self, msg: str):
        self.btn_process.setEnabled(True)
        self.btn_process.setText("Process")
        QtWidgets.QMessageBox.critical(self, "Processing failed", msg)
        self.lbl_metrics.setText("PSNR: - | SSIM: -")

    def _update_metrics(self):
        if self.original is None or self.processed is None:
            self.lbl_metrics.setText("PSNR: - | SSIM: -")
            return
        orig = self.original
        proc = self.processed
        # Ensure same size for metrics
        if orig.shape[:2] != proc.shape[:2]:
            # resize processed to original size (for SR comparisons)
            proc_resized = cv2.resize(proc, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            proc_resized = proc
        # compute on RGB
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        proc_rgb = cv2.cvtColor(proc_resized, cv2.COLOR_BGR2RGB)
        p = psnr(orig_rgb, proc_rgb, data_range=255)
        s = ssim(orig_rgb, proc_rgb, channel_axis=2, data_range=255)
        self.lbl_metrics.setText(f"PSNR: {p:.2f} dB | SSIM: {s:.3f}")

    def save_current(self):
        if self.processed is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Please process an image first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", "output/result.png", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        try:
            imwrite(path, self.processed)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def save_all(self):
        if not self.image_paths:
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder", "output")
        if not out_dir:
            return
        mode = self.mode.currentText()
        strength = self.strength.value() / 100.0
        extra = self.extra.currentText()

        progress = QtWidgets.QProgressDialog("Processing images...", "Cancel", 0, len(self.image_paths), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)

        for i, path in enumerate(self.image_paths):
            if progress.wasCanceled():
                break
            try:
                img = imread(path)
                if mode == "Denoise":
                    out = denoise(img, method=extra or "fastnlm", strength=strength)
                elif mode == "Deblur":
                    out = deblur(img, method=extra or "wiener", strength=strength)
                else:
                    if extra == "4x":
                        scale = 4
                    else:
                        scale = 2
                    out = super_resolve(img, scale=scale)
                name = os.path.splitext(os.path.basename(path))[0]
                ext = ".png"
                save_path = os.path.join(out_dir, f"{name}_{mode.replace(' ', '')}.png")
                imwrite(save_path, out)
            except Exception as e:
                # continue but show message at end
                print(f"Failed {path}: {e}")
            progress.setValue(i + 1)
        progress.close()
        QtWidgets.QMessageBox.information(self, "Done", "Batch processing completed.")


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # Modern Fusion style with dark palette
    QtWidgets.QApplication.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(20, 20, 20))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 220))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(0, 0, 0))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(230, 230, 230))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(64, 128, 255))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
    app.setPalette(palette)
    app.setStyleSheet(
        "QLabel{color:#ddd} QComboBox{background:#2b2b2b; color:#eee; border:1px solid #444; padding:3px} "
        "QSlider::groove:horizontal{height:6px; background:#333; border-radius:3px} "
        "QSlider::handle:horizontal{background:#888; width:14px; margin:-4px 0; border-radius:7px} "
        "QSlider::sub-page:horizontal{background:#5a8dee; border-radius:3px}"
    )
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
