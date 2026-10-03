"""Conservative, configurable image preprocessing. Defaults only resize + RGB."""
from dataclasses import asdict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.config import PreprocessConfig


class ImageLoadError(ValueError):
    pass


def load_image(path) -> Image.Image:
    try:
        img = Image.open(path)
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise ImageLoadError(f"Cannot load image: {e}") from e


class ImagePreprocessor:
    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()

    def describe(self) -> dict:
        return asdict(self.config)

    def __call__(self, image: Image.Image) -> Image.Image:
        c = self.config
        img = image.convert("RGB")
        if c.max_side and max(img.size) > c.max_side:
            img.thumbnail((c.max_side, c.max_side), Image.LANCZOS)
        if c.deskew:
            img = self.deskew(img)
        if c.grayscale:
            img = ImageOps.grayscale(img).convert("RGB")
        if c.contrast:
            img = ImageEnhance.Contrast(img).enhance(c.contrast_factor)
        if c.denoise:
            img = img.filter(ImageFilter.MedianFilter(size=3))
        if c.sharpen:
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))
        return img

    @staticmethod
    def deskew(img: Image.Image, max_angle: float = 10.0) -> Image.Image:
        """Estimate skew from dark pixels (OpenCV). Small corrections only."""
        try:
            import cv2
        except ImportError:
            return img
        gray = np.array(img.convert("L"))
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return img
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
        angle = angle - 90 if angle > 45 else angle
        if abs(angle) < 0.3 or abs(angle) > max_angle:
            return img
        return img.rotate(angle, expand=True, fillcolor="white")
