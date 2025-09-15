import argparse
from math import ceil, log, sqrt
from pathlib import Path

import numpy as np
from PIL import Image


def factor(n: int) -> list[int]:
    """Factors a number with trial division."""
    factors = {1, n}
    for i in range(2, ceil(sqrt(n)) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return sorted(factors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("out"))
    parser.add_argument("--dest", type=Path, default=Path("grid.png"))
    parser.add_argument("--format", type=str, default="png")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()

    width, height = args.width, args.height
    for path, _, files in args.outdir.walk():
        files = sorted(files)
        n = len(files)
        assert n > 0, f"No images in {args.outdir}."
        if width is not None:
            assert n % width == 0, f"{n} not divisible by {width}."
        if height is not None:
            assert n % height == 0, f"{n} not divisible by {height}."
        if width is not None and height is not None:
            assert n == width * height, f"{n} not {width} x {height}."
        width = min(factor(n), key=lambda m: (abs(log(n // m) - log(m)), -m))
        height = n // width
        images = [Image.open(path / file) for file in files]
        x = np.stack([np.asarray(image) for image in images])
        _, h, w, c = x.shape
        x = x.reshape(height, width, *x.shape[1:]).transpose(0, 2, 1, 3, 4)
        x = x.reshape(height * h, width * w, c)
        Image.fromarray(x).save(args.dest, format=args.format)
