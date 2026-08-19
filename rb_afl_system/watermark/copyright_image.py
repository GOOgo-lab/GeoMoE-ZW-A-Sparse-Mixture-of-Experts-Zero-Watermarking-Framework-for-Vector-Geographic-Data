"""Load a copyright image as a deterministic binary watermark."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _arnold_square(bits: np.ndarray, iterations: int, inverse: bool = False) -> np.ndarray:
    side = int(round(math.sqrt(bits.size)))
    if side * side != bits.size:
        return bits.copy()
    image = bits.reshape(side, side).copy()
    for _ in range(max(0, int(iterations))):
        result = np.empty_like(image)
        for x in range(side):
            for y in range(side):
                if inverse:
                    nx, ny = (2 * x - y) % side, (-x + y) % side
                else:
                    nx, ny = (x + y) % side, (x + 2 * y) % side
                result[nx, ny] = image[x, y]
        image = result
    return image.reshape(-1)


def load_copyright_bits(
    path: str | Path,
    bit_length: int = 256,
    threshold: int = 128,
    arnold_iterations: int = 0,
) -> np.ndarray:
    """Resize a grayscale image to ``bit_length`` pixels and binarize it."""
    from PIL import Image

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Copyright image not found: {image_path}")
    side = int(round(math.sqrt(int(bit_length))))
    if side * side == int(bit_length):
        size = (side, side)
    else:
        size = (int(bit_length), 1)
    with Image.open(image_path) as image:
        gray = image.convert("L").resize(size, Image.Resampling.LANCZOS)
        bits = (np.asarray(gray, dtype=np.uint8).reshape(-1) >= int(threshold)).astype(np.uint8)
    return _arnold_square(bits, arnold_iterations, inverse=False)


def inverse_arnold_bits(bits: np.ndarray, iterations: int) -> np.ndarray:
    return _arnold_square(np.asarray(bits, dtype=np.uint8).reshape(-1), iterations, inverse=True)
