#!/usr/bin/env python3
"""Generate FontSrcTarget dataset from a folder of pre-rendered glyph images.

Instead of rendering target glyphs from a TTF/OTF file, this script reads
glyph images directly from a folder (e.g., AI-generated glyphs). The source
glyphs are still rendered from a source font.

Glyph folder layout:  {character}.png  (e.g., 万.png, 鶴.png)
Each image should be 256x256 RGB.

The user specifies --train-count; remaining glyphs go to test.
References for both train and test are drawn from the train set.

Output layout (same as generate_font_dataset.py):
    <output-dir>/train/001_<font-name>/  (1024x256 JPEGs + metadata.json)
    <output-dir>/test/001_<font-name>/   (1024x256 JPEGs + metadata.json)

Usage:
    python scripts/generate_glyph_dataset.py \
        --source-font data/test_fonts/思源宋体light.otf \
        --glyph-dir data/sample_glyphs \
        --output-dir data/glyph_dataset \
        --train-count 200 \
        --font-name my_custom_font

    # Train only
    python scripts/generate_glyph_dataset.py \
        --source-font data/test_fonts/思源宋体light.otf \
        --glyph-dir data/sample_glyphs \
        --output-dir data/glyph_dataset \
        --train-count 200 \
        --train-only
"""
import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
import shutil
from typing import List, Optional
import os
import zipfile
from font2data.FontData import FontData
import pyarrow.feather as ft
import sys
logger = logging.getLogger(__name__)

from PIL import Image

# 讓腳本從專案根目錄找到 data_processing
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data_processing.font_utils import GlyphRenderer
from data_processing.pipeline import (
    create_combined_image,
    _format_codepoint,
    _filename_for_codepoint,
    _pick_refs,
    create_test_npz,
)


DEFAULT_JPEG_QUALITY = 95


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FontSrcTarget dataset from pre-rendered glyph images."
    )
    parser.add_argument("--source-font", required=True, help="Path to source/reference font (TTF/OTF).")
    parser.add_argument("--glyph-dir", required=True, help="Directory containing glyph images ({char}.png).")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory. Creates train/ and test/ subdirectories.")
    parser.add_argument("--train-count", required=True, type=int,
                        help="Number of glyphs for training. Remaining go to test.")
    parser.add_argument("--font-name", type=str, default=None,
                        help="Font name for output folder (default: glyph-dir folder name).")
    parser.add_argument("--font-index", type=int, default=1, help="Font index for folder naming (default: 1).")
    parser.add_argument("--resolution", type=int, default=256, help="Glyph resolution (default: 256).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split (default: 42).")
    parser.add_argument("--train-only", action="store_true", help="Generate train dataset only, skip test.")
    parser.add_argument("--input-path", type=str, default=None, help="Input path for glyphs (default: None).")
    args = parser.parse_args()

    # if not Path(args.glyph_dir).is_dir():
    #     parser.error(f"--glyph-dir does not exist: {args.glyph_dir}")
    if not (Path(args.source_font).is_file() or os.path.isdir(args.source_font)):
        parser.error(f"--source-font does not exist: {args.source_font}")

    return args


def load_glyphs(glyph_dir: Path) -> list:
    """Load glyph images from folder. Returns sorted list of (codepoint, path)."""
    glyphs = []
    for f in glyph_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        char = f.stem
        if len(char) != 1:
            continue
        codepoint = ord(char)
        glyphs.append((codepoint, f))
    glyphs.sort(key=lambda x: x[0])
    return glyphs


def create_ref_grid_from_images(
    glyph_map: dict,
    ref_codepoints: List[int],
    cell_size: int = 128,
) -> Optional[Image.Image]:
    """Create a 256x256 reference grid from 4 glyph images."""
    grid = Image.new("RGB", (256, 256), (255, 255, 255))
    positions = [(0, 0), (cell_size, 0), (0, cell_size), (cell_size, cell_size)]
    for idx, cp in enumerate(ref_codepoints):
        img = Image.open(glyph_map[cp]).convert("RGB")
        img = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(img, positions[idx])
    return grid


def generate_split(
    split_name: str,
    codepoints: List[int],
    ref_pool: List[int],
    glyph_map: dict,
    source_font_data: FontData,
    #source_renderer: GlyphRenderer,
    output_dir: Path,
    font_name: str,
    font_index: int,
    source_font_path: Path,
    resolution: int,
    seed: int,
) -> dict:
    """Generate train or test split."""
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    characters = []

    for local_index, cp in enumerate(codepoints):
        source_img = source_font_data.char2img(chr(cp))
        # source_img = source_renderer.render(cp)
        if source_img is None:
            failed += 1
            continue

        target_img = Image.open(glyph_map[cp]).convert("RGB")
        if target_img.size != (resolution, resolution):
            target_img = target_img.resize((resolution, resolution), Image.LANCZOS)

        refs = _pick_refs(ref_pool, cp, seed)
        if refs is None:
            failed += 1
            continue

        ref_grid_1 = create_ref_grid_from_images(glyph_map, refs[:4])
        ref_grid_2 = create_ref_grid_from_images(glyph_map, refs[4:])

        combined = create_combined_image(source_img, target_img, ref_grid_1, ref_grid_2)
        filename = _filename_for_codepoint(cp, local_index)
        combined.save(output_dir / filename, "JPEG", quality=DEFAULT_JPEG_QUALITY)

        characters.append({
            "codepoint": _format_codepoint(cp),
            "character": chr(cp),
            "filename": filename,
            "reference_codepoints_1": [_format_codepoint(r) for r in refs[:4]],
            "reference_codepoints_2": [_format_codepoint(r) for r in refs[4:]],
        })
        successful += 1
    logger.info("  %s: %d extracted, %d failed", split_name, successful, failed)

    metadata = {
        "font_name": font_name,
        "font_index": font_index,
        "source_font": str(source_font_path),
        "glyph_source": "image_folder",
        "dataset_type": split_name,
        "extraction_date": datetime.now().isoformat(),
        "sample_requested": len(codepoints),
        "extracted_count": successful,
        "failed_count": failed,
        "resolution": resolution,
        "image_dimensions": "1024x256",
        "layout": "source(256) + target(256) + refs_grid1(256) + refs_grid2(256)",
        "characters": characters,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {"success": True, "extracted": successful, "failed": failed}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args()

    args.glyph_dir = extract_glyphs(args)
    print(f"Glyph directory: {args.glyph_dir}")
    glyph_dir = Path(args.glyph_dir)
    output_dir = Path(args.output_dir)
    source_font_path = Path(args.source_font)
    font_name = args.font_name or glyph_dir.name
    font_index = args.font_index
    folder_name = f"{font_index:03d}_{font_name}"

    # Load glyphs
    print(f"Loading glyphs from {glyph_dir}")
    glyphs = load_glyphs(glyph_dir)
    glyph_map = {cp: path for cp, path in glyphs}
    all_codepoints = [cp for cp, _ in glyphs]
    total = len(all_codepoints)

    if args.train_count >= total:
        print(f"Error: --train-count ({args.train_count}) >= total glyphs ({total}). "
              f"Need at least 1 for test.")

        if args.train_only:
            args.train_count = total
        
        print("Use 95% of the glyphs for train, and 5% for test")
        args.train_count = int(total * 0.95)
        print(f"New train count: {args.train_count}")



    if args.train_count < 9:
        print(f"Error: --train-count ({args.train_count}) must be >= 9 for references.")
        return

    # Shuffle and split
    rng = random.Random(args.seed)
    shuffled = list(all_codepoints)
    rng.shuffle(shuffled)
    train_cps = sorted(shuffled[:args.train_count])
    test_cps = sorted(shuffled[args.train_count:])

    print(f"Glyphs: {total} total, {len(train_cps)} train, {len(test_cps)} test")

    source_font_data = FontData(args.source_font, debug=True, font_size=args.resolution)
    # # Source renderer
    # source_renderer = GlyphRenderer(str(source_font_path), args.resolution)

    # Generate train
    train_out = output_dir / "train" / folder_name
    print(f"\n=== Generating train -> {train_out} ===")
    train_result = generate_split(
        "train", train_cps, train_cps, glyph_map, source_font_data,
        train_out, font_name, font_index, source_font_path, args.resolution, args.seed,
    )
    print(f"  extracted={train_result['extracted']} failed={train_result['failed']}")

    # Generate test
    if not args.train_only:
        test_out = output_dir / "test" / folder_name
        print(f"\n=== Generating test -> {test_out} ===")
        test_result = generate_split(
            "test", test_cps, train_cps, glyph_map, source_font_data,
            test_out, font_name, font_index, source_font_path, args.resolution, args.seed,
        )
        print(f"  extracted={test_result['extracted']} failed={test_result['failed']}")

        # Convert test set to NPZ
        test_dir = output_dir / "test"
        npz_path = output_dir / "test.npz"
        print(f"\n=== Creating test NPZ -> {npz_path} ===")
        result = create_test_npz(test_dir, npz_path)
        print(f"  {result['samples']} samples ({result['file_size_mb']:.1f} MB)")

    print("\nDone!")



def extract_glyphs(args: argparse.Namespace):
    TEMP_WORK_DIR = "data/sample_glyphs"
    if os.path.isfile(args.glyph_dir):
        print(f"It's a file, Extracting {args.glyph_dir} to {TEMP_WORK_DIR}")
        # clear TEMP_WORK_DIR
        if os.path.exists(TEMP_WORK_DIR):
            shutil.rmtree(TEMP_WORK_DIR)
        os.makedirs(TEMP_WORK_DIR, exist_ok=True)
        
        if os.path.splitext(args.glyph_dir)[1] == ".zip":
            print(f"It's a zip file, Extracting {args.glyph_dir} to {TEMP_WORK_DIR}")
            with zipfile.ZipFile(args.glyph_dir, 'r') as zip_ref:
                zip_ref.extractall(TEMP_WORK_DIR)

        if os.path.splitext(args.glyph_dir)[1] == ".arrow":
            print(f"It's a arrow file")
            df = ft.read_table(args.glyph_dir).to_pandas()
            out_dir = Path(TEMP_WORK_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, row in df.iterrows():
                char=row['char']
                out_path = out_dir / f"{char}.png"
                out_path.write_bytes(row['images'])  
            print(f"Saved {len(df)} images to: {out_dir.resolve()}")

        return TEMP_WORK_DIR
    else:
        return args.glyph_dir



if __name__ == "__main__":
    main()
