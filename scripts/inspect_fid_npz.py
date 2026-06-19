#!/usr/bin/env python3
"""
Inspect FID statistics .npz file: list keys, shapes, dtypes, and show 3 sample values.
Usage (with uv): uv run python scripts/inspect_fid_npz.py [npz_path]
"""

import argparse
import os

import numpy as np

DEFAULT_NPZ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fid_stats",
    "font_100_test_chars_fid_stats.npz",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect .npz file (keys, shapes, sample data).")
    parser.add_argument(
        "npz",
        nargs="?",
        default=DEFAULT_NPZ,
        help=f"Path to .npz file (default: {DEFAULT_NPZ})",
    )
    return parser.parse_args()


def sample_array(arr, max_sample=3):
    """Return a small sample of array for display."""
    if arr.ndim == 0:
        return arr.item()
    if arr.ndim == 1:
        return arr[:max_sample]
    if arr.ndim == 2:
        return arr[:max_sample, :max_sample]
    # higher dim: slice first two dims
    return np.array(arr[:max_sample])


def main():
    args = parse_args()
    npz_path = os.path.abspath(args.npz)

    if not os.path.exists(npz_path):
        print(f"File not found: {npz_path}")
        return

    print(f"Loading: {npz_path}\n")
    data = np.load(npz_path, allow_pickle=True)

    print("=== Keys in .npz ===")
    keys = list(data.keys())
    print(keys)
    print()

    print("=== Each key: shape, dtype ===")
    for key in keys:
        arr = data[key]
        print(f"  {key!r}: shape={arr.shape}, dtype={arr.dtype}")

    print("\n=== Sample data (first 3 elements or 3x3 slice) ===")
    for key in keys:
        arr = data[key]
        samp = sample_array(arr, max_sample=3)
        print(f"\n  {key!r}:")
        if isinstance(samp, np.ndarray):
            print(samp)
        else:
            print(f"    {samp}")

    data.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
