#!/usr/bin/env python3
"""Run inference with sensible defaults for generate_chars.py."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import shutil
import subprocess
import sys
import zipfile

INFERENCE_FLAGS = (
  "--train_dir",
  "--source_font",
  "--npz_file",
  "--generate_chars",
  "--checkpoint",
  "--output_dir",
  "--glyph-dir",
  "--train-count",
  "--dataset-output-dir",
  "--matmul_precision",
)

EPILOG = """\
Examples:
  python inference.py --checkpoint run/my/checkpoint-last.pth
  python inference.py --source_font data/HanaMinA.otf --train-count 500
  python inference.py --dataset-output-dir data/my_glyph_dataset --train_dir data/my_glyph_dataset/train
"""

def zip_generated_images(input_dir, output_dir):

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    images = os.listdir(input_dir)
    with zipfile.ZipFile(f"{output_dir}", "w") as zipf:
        for image in images:
            zipf.write(f"{input_dir}/{image}", image)
    print(f"zip generated images to {output_dir}")

    # # Copy the zip file to new 
    # base_path = output_dir
    # base_name, ext = os.path.splitext(base_path)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # new_output_dir = f"{base_name}_{timestamp}{ext}"
    # shutil.copy(output_dir, new_output_dir)
    # print(f"copy zip file to {new_output_dir}")


def _reject_equals_form(argv: list[str]) -> None:
  for arg in argv:
    for flag in INFERENCE_FLAGS:
      if arg.startswith(f"{flag}="):
        print(
          f"Error: please use '{flag} value' (space-separated), not '{flag}=value': {arg}",
          file=sys.stderr,
        )
        sys.exit(2)


def generate_glyph_dataset(
  source_font: str,
  glyph_dir: str,
  train_count: int,
  output_dir: str = "data/sample_glyph_dataset",
) -> None:
  print(f"Generating glyph dataset from {source_font} with {train_count} glyphs...")
  p = subprocess.run(
    [
      sys.executable,
      "scripts/generate_glyph_dataset_from_a_file.py",
      "--source-font",
      source_font,
      "--glyph-dir",
      glyph_dir,
      "--output-dir",
      output_dir,
      "--train-count",
      str(train_count),
    ],
    check=True,
  )
  return p.returncode

def create_inference_npz(
  train_dir: str,
  source_font: str,
  npz_file: str,
  generate_chars: str,
) -> None:
  print(f"Creating inference .npz from {train_dir} with {source_font} and {generate_chars}...")
  p = subprocess.run(
    [
      sys.executable,
      "scripts/create_inference_npz.py",
      "--train_dir",
      train_dir,
      "--source_font",
      source_font,
      "--output",
      npz_file,
      "--charset",
      generate_chars,
      "--font_indices",
      "0",
    ],
    check=True,
  )
  return p.returncode




def generate_chars(
  checkpoint: str,
  npz_file: str,
  output_dir: str,
  sampling_method: str = "ab2",
  num_sampling_steps: int = 20,
  matmul_precision: str = "high",
  batch_size: int = 64,
  content_preserving_cfg: bool = True,
) -> None:
  print(f"Generating characters from {checkpoint} with {npz_file} to {output_dir}...")
  cmd = [
      sys.executable,
      "generate_chars.py",
      "--checkpoint",
      checkpoint,
      "--test_npz",
      npz_file,
      "--output_dir",
      output_dir,
      "--matmul_precision",
      matmul_precision,
      "--batch_size",
      str(batch_size),
      "--sampling_method",
      sampling_method,
      "--num_sampling_steps",
      str(num_sampling_steps),
  ]
  if content_preserving_cfg:
    cmd.append("--content_preserving_cfg")
  p = subprocess.run(cmd, check=True)
  return p.returncode

def main() -> None:
  _reject_equals_form(sys.argv[1:])

  parser = argparse.ArgumentParser(
    description="Run inference with sensible defaults for generate_chars.py.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=EPILOG,
  )

  parser.add_argument("--source_font", default="data/1")
  parser.add_argument("--generate_chars", default="basic")
  parser.add_argument("--checkpoint", default="run/checkpoint-last.pth")
  parser.add_argument("--output_dir", default="run/generated_chars")
  parser.add_argument("--glyph-dir", default="data/ref.arrow")
  parser.add_argument("--train-count", default=200)
  parser.add_argument("--npz_file", default="run/glyph_dataset/inference.npz")
  parser.add_argument("--dataset-output-dir", default="run/sample_glyph_dataset")
  parser.add_argument("--zip_output", action="store_true")
  parser.add_argument("--batch_size", default=64)
  parser.add_argument(
    "--matmul_precision",
    default="high",
    choices=["highest", "high", "medium"],
    help="Float32 matmul precision for CUDA generation (TF32 when high).",
  )
  parser.add_argument(
    "--content_preserving_cfg",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Keep source glyph structure during CFG (default: on).",
  )

  parser.add_argument('--batch_number', type=int, default=None,
                        help='Batch number for zip output (default: None)')
                        
  args = parser.parse_args()

  # Step 0: Check if the dataset is already generated
  if not os.path.exists(args.dataset_output_dir):
    print(f"Dataset not found at {args.dataset_output_dir}, generating...")
    # Step 1: Generate glyph dataset
    result = generate_glyph_dataset(
      args.source_font,
      args.glyph_dir,
      args.train_count,
      output_dir=args.dataset_output_dir,
    )
    if result != 0:
      print(f"Failed to generate glyph dataset, return code: {result}")
      exit(1)
  else:
    print(f"Dataset found at {args.dataset_output_dir}, using existing dataset...")

  train_dir = os.path.join(args.dataset_output_dir, "train")

  # Step 2: Create inference .npz
  result = create_inference_npz(
    train_dir,
    args.source_font,
    args.npz_file,
    args.generate_chars,
  )
  if result != 0:
    print(f"Failed to create inference .npz, return code: {result}")
    exit(1)

  print(f"Try make temp directory")
  TEMP_DIR = f"temp_generated_chars"
  os.makedirs(TEMP_DIR, exist_ok=True)
  OUTPUT_DIR = args.output_dir if not args.zip_output else TEMP_DIR
  print(f"OUTPUT_DIR: {OUTPUT_DIR}")
  # Step 3: Generate characters
  result = generate_chars(
    checkpoint=args.checkpoint, 
    npz_file=args.npz_file, 
    output_dir=OUTPUT_DIR, 
    matmul_precision=args.matmul_precision, 
    batch_size=args.batch_size,
    content_preserving_cfg=args.content_preserving_cfg,
  )

  if result != 0:
    print(f"Failed to generate characters, return code: {result}")
    exit(1)
  # Step 4: Zip generated images
  if args.zip_output:
    try:
        zip_file_name = f"{args.output_dir}/generated_images.zip" \
          if args.batch_number is None else f"{args.output_dir}/generated_images_batch_{args.batch_number}.zip"
        zip_generated_images(OUTPUT_DIR, zip_file_name)
    finally:
        shutil.rmtree(OUTPUT_DIR)

if __name__ == "__main__":

  main()
