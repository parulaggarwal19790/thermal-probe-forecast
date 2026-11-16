"""Write a tiny synthetic dataset in the expected format, to check the pipeline runs end to end.

Each board gets two growing hot spots and one fading one at random probe positions, plus a laser
trace and noise. The labels are derived from those spots. It is far simpler than real thermal data:
use it to try the code, not to judge the model.

Usage:
    python make_toy_data.py toy_data
    python train.py toy_data toy_pred.csv --arch convnextv2_atto.fcmae_ft_in1k --epochs 2 --folds 2 --full-models 0
    python evaluate.py toy_data/test_labels.csv toy_pred.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

PROBE_COLUMNS = [18, 42, 66, 90, 114, 138]  # sensor columns; 4 pixels per column


def colour(v):
    """Blue -> red -> yellow colour map for intensities in [0, 1]."""
    r = np.clip(2 * v, 0, 1)
    g = np.clip(2 * v - 1, 0, 1)
    b = np.clip(1 - 2 * v, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def board(rng):
    growing = rng.choice(6, 2, replace=False)
    fading = rng.integers(6)
    rates = rng.uniform(0.5, 1.0, 2)
    y, x = np.mgrid[0:160, 0:640]
    strips = []
    for t in range(4):
        v = rng.normal(0.15, 0.04, (160, 640))
        for p, rate in zip(growing, rates):
            cx, cy = PROBE_COLUMNS[p] * 4 + 2, 80
            v += rate * (t + 1) / 4 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 14 ** 2))
        cx = PROBE_COLUMNS[fading] * 4 + 2
        v += 0.8 * (1 - t / 4) * np.exp(-((x - cx) ** 2 + (y - 50) ** 2) / (2 * 12 ** 2))
        row = rng.integers(10, 150)
        v[row - 2:row + 3, rng.integers(0, 200):rng.integers(400, 640)] += 0.6  # a laser trace
        strips.append(colour(np.clip(v / v.max(), 0, 1)))
    # future levels 0-4 at three later times; the fading spot counts for nothing
    matrix = np.zeros((3, 6), int)
    for p, rate in zip(growing, rates):
        for k in range(3):
            matrix[k, p] = min(4, int(round(rate * (k + 2) * 1.2)))
    gain = matrix.sum(0)
    pair = sorted(np.argsort(-gain, kind="stable")[:2].tolist())
    return np.concatenate(strips, 0), matrix, pair


def main(out_dir, n_train=96, n_test=24, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    rows = []
    for i in range(n_train + n_test):
        img, matrix, pair = board(rng)
        path = f"images/tp_{i:05d}.jpg"
        Image.fromarray(img).save(os.path.join(out_dir, path), quality=90)
        rows.append({"case_id": f"tp_{i:05d}", "image_path": path,
                     "response_matrix": json.dumps(matrix.tolist(), separators=(",", ":")),
                     "probe_pair": f"{pair[0]}>{pair[1]}"})
    df = pd.DataFrame(rows)
    df.iloc[:n_train].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df.iloc[n_train:][["case_id", "image_path"]].to_csv(os.path.join(out_dir, "test.csv"), index=False)
    df.iloc[n_train:].drop(columns="image_path").to_csv(os.path.join(out_dir, "test_labels.csv"), index=False)
    print(f"wrote {n_train} training and {n_test} test boards to {out_dir}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "toy_data")
