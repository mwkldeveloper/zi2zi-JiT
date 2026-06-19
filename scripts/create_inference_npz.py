"""Create inference .npz for characters without ground-truth targets.

Reads a training dataset folder for style references and font indices,
renders source glyphs from a source font, and covers the full charset
(default: GB2312) minus training characters.

Output npz keys
---------------
  font_labels    : (N,)             int64   font class indices
  char_labels    : (N,)             int64   character class indices (sequential)
  unicode_labels : (N,)             int64   Unicode codepoints
  content_images : (N, 3, 256, 256) uint8   source glyph (rendered from source font)
  style_images   : (N, 3, 128, 128) uint8   reference glyph (from train dataset)

No target_images key — targets do not exist yet.

Usage
-----
  # Generate NPZ covering full GB2312 minus train chars, for all fonts in train_dir
  python scripts/create_inference_npz.py \
      --train_dir data/sample_dataset/train \
      --source_font data/test_fonts/思源宋体light.otf \
      --output data/sample_dataset/inference.npz

  # Limit to specific fonts
  python scripts/create_inference_npz.py \
      --train_dir data/glyph_dataset/train \
      --source_font data/test_fonts/思源宋体light.otf \
      --output data/glyph_dataset/inference.npz \
      --font_indices 0
"""

import argparse
import json
import os
import random
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_processing.extra_charsets import get_charset_codepoints
from data_processing.font_utils import GlyphRenderer, load_font, get_cjk_codepoints

from font2data.FontData import FontData


def parse_args():
    parser = argparse.ArgumentParser(description="Create inference .npz (no targets) from train dataset + source font")
    parser.add_argument("--train_dir", required=True, type=str,
                        help="Train dataset directory (contains {idx}_{name}/ folders with metadata.json).")
    parser.add_argument("--source_font", required=True, type=str,
                        help="Path to source font (TTF/OTF) for rendering content images.")
    parser.add_argument("--output", default="inference.npz", type=str,
                        help="Output .npz file path.")
    parser.add_argument("--charset", default="gb2312", type=str, nargs='+',
                        # choices=["gb2312", "gbk", "big5", "jisx0208", "ksx1001","basic", "extA", "extB", "extC", "extD", "extE", "extF", "extG", "extH", "extI", "extJ"], # <-----增加更多类别
                        help="Target charset to cover (default: gb2312).")
    parser.add_argument("--max_chars", default=None, type=int,
                        help="Max characters to include per font (default: all). Takes first N from sorted charset.")
    parser.add_argument("--ref_index", default=0, type=int, choices=range(8),
                        help="Which of the 8 reference glyphs to use (0-7). Default: 0.")
    parser.add_argument("--ref_size", default=128, type=int,
                        help="Reference image size. Default: 128.")
    parser.add_argument("--resolution", default=256, type=int,
                        help="Content image resolution. Default: 256.")
    parser.add_argument("--font_indices", default=None, type=str,
                        help="Comma-separated font indices (0-based) to include. Default: all.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for reference selection.")
    parser.add_argument("--pad_to", default=None, type=int,
                        help="Pad output to exactly this many samples by repeating cyclically.")
    return parser.parse_args()


def extract_ref(img, ref_global_idx, ref_size):
    """Extract a single 128x128 reference glyph from the two 256x256 grids."""
    grid_idx = ref_global_idx // 4
    ref_idx = ref_global_idx % 4
    grid_x = 512 if grid_idx == 0 else 768
    row = ref_idx // 2
    col = ref_idx % 2
    x1 = grid_x + col * 128
    y1 = row * 128
    ref = img.crop((x1, y1, x1 + 128, y1 + 128))
    if ref.size[0] != ref_size or ref.size[1] != ref_size:
        ref = ref.resize((ref_size, ref_size), Image.LANCZOS)
    return ref


def load_train_info(train_dir):
    """Load font folders: returns list of (font_idx, font_folder_path, train_codepoints, ref_images)."""
    fonts = []
    font_dirs = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')
    ])

    for font_name in font_dirs:
        font_path = os.path.join(train_dir, font_name)
        idx_str = font_name.split('_')[0]
        if idx_str.startswith("'"):
            idx_str = idx_str[1:]
        font_idx = int(idx_str) - 1

        # Load metadata for train codepoints
        meta_path = os.path.join(font_path, 'metadata.json')
        if not os.path.exists(meta_path):
            print(f"Warning: no metadata.json in {font_path}, skipping")
            continue

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        train_codepoints = set()
        for char_info in meta.get('characters', []):
            cp_str = char_info['codepoint']
            if cp_str.startswith('U+'):
                train_codepoints.add(int(cp_str[2:], 16))

        # Collect training image paths for reference extraction
        train_images = {}
        for filename in os.listdir(font_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            parts = filename.split('_')
            for part in parts[1:]:
                part_clean = os.path.splitext(part)[0]
                if part_clean.startswith('U+'):
                    cp = int(part_clean[2:], 16)
                    train_images[cp] = os.path.join(font_path, filename)
                    break

        fonts.append({
            'font_idx': font_idx,
            'font_name': font_name,
            'font_path': font_path,
            'train_codepoints': train_codepoints,
            'train_images': train_images,
        })

    return fonts


def main():
    args = parse_args()

    if not os.path.isdir(args.train_dir):
        raise FileNotFoundError(f"--train_dir does not exist: {args.train_dir}")
    if not (os.path.isfile(args.source_font) or os.path.isdir(args.source_font)):
        raise FileNotFoundError(f"--source_font does not exist: {args.source_font}")

    # Load charset
    charset_cps = get_charset_codepoints(args.charset)
    print(f"Charset {args.charset}: {len(charset_cps)} codepoints")

    # Filter to codepoints the source font can render
    source_font_data = FontData(args.source_font, debug=True, font_size=args.resolution)
    source_cps = source_font_data.get_supported_chars()
    # print(f"chars: {chars}")
    # source_font, source_path = load_font(args.source_font)
    # source_cps = get_cjk_codepoints(source_font)
    print(f"source_cps: {len(source_cps)}")
    # find intersection of charset_cps and source_cps
    renderable_cps = charset_cps & source_cps 
    print(f"Source font can render: {len(renderable_cps)} / {len(charset_cps)}")

    # Load train info
    fonts = load_train_info(args.train_dir)
    print(f"Found {len(fonts)} font(s) in train dir")

    # Filter fonts if requested
    if args.font_indices is not None:
        selected = set(int(x.strip()) for x in args.font_indices.split(','))
        fonts = [f for f in fonts if f['font_idx'] in selected]
        print(f"Filtered to {len(fonts)} font(s): {[f['font_idx'] for f in fonts]}")

    # Build samples: (font_idx, codepoint) for each font
    samples = []
    for font_info in fonts:

        # # use train_codepoints to filter out the codepoints that are already in the train dataset
        # print(f"font_info: {font_info}")
        # train_cps = font_info['train_codepoints']
        # print(f"train_cps: {train_cps}")
        # target_cps = sorted(renderable_cps - train_cps)
        # print(f"target_cps: {target_cps}")
        # if args.max_chars is not None:
        #     target_cps = target_cps[:args.max_chars]
        # for cp in target_cps:
        for cp in renderable_cps:
            samples.append((font_info, cp))

    n = len(samples)
    if n == 0:
        print("Error: no samples to generate")
        return
    print(f"Total samples: {n}")

    # Source renderer
    #source_renderer = GlyphRenderer(str(source_path), args.resolution)

    # Allocate arrays
    font_labels = np.empty(n, dtype=np.int64)
    char_labels = np.empty(n, dtype=np.int64)
    unicode_labels = np.empty(n, dtype=np.int64)
    content_images = np.empty((n, 3, args.resolution, args.resolution), dtype=np.uint8)
    target_images = np.full((n, 3, args.resolution, args.resolution), 255, dtype=np.uint8)
    style_images = np.empty((n, 3, args.ref_size, args.ref_size), dtype=np.uint8)

    rng = random.Random(args.seed)
    skipped = 0

    for i, (font_info, cp) in enumerate(tqdm(samples, desc="Processing")):
        # Render source/content image
        #print(f"cp: {cp}")
        # src_img = source_renderer.render(cp)
        src_img = source_font_data.char2img(chr(cp))
        #print(f"src_img: {src_img}")
        if src_img is None:
            # White placeholder
            content_images[i] = 255
            skipped += 1
        else:
            content_images[i] = np.array(src_img).transpose(2, 0, 1)

        # Pick a random training image for style reference
        train_imgs = font_info['train_images']
        if train_imgs:
            ref_cp = rng.choice(list(train_imgs.keys()))
            ref_path = train_imgs[ref_cp]
            ref_full_img = Image.open(ref_path).convert('RGB')
            ref_img = extract_ref(ref_full_img, args.ref_index, args.ref_size)
            style_images[i] = np.array(ref_img).transpose(2, 0, 1)
        else:
            style_images[i] = 255

        font_labels[i] = font_info['font_idx']
        char_labels[i] = i
        unicode_labels[i] = cp

    if skipped > 0:
        print(f"Warning: {skipped} source glyphs could not be rendered (white placeholder)")

    num_original = n
    if args.pad_to is not None and args.pad_to > n:
        pad_n = args.pad_to
        idx = np.arange(pad_n) % n
        font_labels = font_labels[idx]
        char_labels = char_labels[idx]
        unicode_labels = unicode_labels[idx]
        content_images = content_images[idx]
        target_images = target_images[idx]
        style_images = style_images[idx]
        print(f"Padded {n} -> {pad_n} samples (cyclic repeat)")
        n = pad_n

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez_compressed(
        args.output,
        font_labels=font_labels,
        char_labels=char_labels,
        unicode_labels=unicode_labels,
        content_images=content_images,
        target_images=target_images,
        style_images=style_images,
        num_original_samples=np.int64(num_original),
    )

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved {n} samples to {args.output} ({file_size:.1f} MB)")
    return


if __name__ == "__main__":
    main()
