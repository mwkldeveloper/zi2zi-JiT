#!/usr/bin/env python3
"""Fine-tune JiT with sensible defaults for lora_single_gpu_finetune_jit.py."""

from __future__ import annotations

import argparse
import subprocess
import sys

DATASET_FLAGS = (
  "--source-font",
  "--glyph-dir",
  "--train-count",
  "--dataset-output-dir",
  "--output_dir",
)

EPILOG = """\
Examples:
  python finetune.py --epochs 50 --batch_size 8 --blr 1e-4
  python finetune.py --data_path data/my/train --output_dir run/my_exp
  python finetune.py --matmul_precision high
  python finetune.py --source-font data/HanaMinA.otf --glyph-dir data/ref.arrow --train-count 500
  python finetune.py --dataset-output-dir data/my_glyph_dataset --train-count 500

Notes:
  - Dataset generation args (--source-font, --glyph-dir, --train-count) are consumed
    by this script and not passed through to lora_single_gpu_finetune_jit.py.
  - Any other args are appended at the end of the finetune command, so they override
    defaults when the same flag appears multiple times.
"""


def _reject_equals_form(argv: list[str]) -> None:
  for arg in argv:
    for flag in DATASET_FLAGS:
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
  output_dir: str,
) -> None:
  subprocess.run(
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


def finetune_model(
  passthru: list[str],
  dataset_dir: str,
  output_dir: str,
  epochs: int,
  base_checkpoint: str,
  model: str,
  batch_size: int,
  save_last_freq: int,
  eval_freq: int,
  sampling_method: str,
  num_sampling_steps: int,
) -> None:
  cfg = 2.4 if model == "JiT-L/16" else 2.6 
  subprocess.run(
    [
      sys.executable,
      "lora_single_gpu_finetune_jit.py",
      "--data_path",
      f"./{dataset_dir}/train/",
      "--test_npz_path",
      f"./{dataset_dir}/test.npz",
      "--output_dir",
      output_dir,
      "--base_checkpoint",
      base_checkpoint,
      "--model",
      model,
      "--num_fonts",
      "1000",
      "--num_chars",
      "20000",
      "--max_chars_per_font",
      "200",
      "--img_size",
      "256",
      "--lora_r",
      "32",
      "--lora_alpha",
      "32",
      "--lora_targets",
      "qkv,proj,w12,w3",
      "--epochs",
      str(epochs),
      "--batch_size",
      str(batch_size),
      "--blr",
      "8e-4",
      "--warmup_epochs",
      "1",
      "--save_last_freq",
      str(save_last_freq),
      "--proj_dropout",
      "0.1",
      "--P_mean",
      "-0.8",
      "--P_std",
      "0.8",
      "--noise_scale",
      "1.0",
      "--cfg",
      str(cfg),
      "--online_eval",
      "--eval_step_folders",
      "--eval_freq",
      str(eval_freq),
      "--gen_bsz",
      "16",
      "--num_images",
      "400",
      "--seed",
      "42",
      "--sampling_method",
      sampling_method,
      "--num_sampling_steps",
      str(num_sampling_steps),
      *passthru,
    ],
    check=True,
  )


def main() -> None:
  _reject_equals_form(sys.argv[1:])

  parser = argparse.ArgumentParser(
    description="Fine-tune JiT with sensible defaults for lora_single_gpu_finetune_jit.py.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=EPILOG,
  )
  parser.add_argument("--source-font", default="data/HanaMinA.otf")
  parser.add_argument("--glyph-dir", default="data/ref.arrow")
  parser.add_argument("--train-count", default=200)
  parser.add_argument("--dataset-output-dir", default="run/sample_glyph_dataset")
  parser.add_argument("--output_dir", default="run/zi2zi-jit_output/")
  parser.add_argument("--epochs", default=200)
  parser.add_argument("--base_checkpoint", default="models/zi2zi-JiT-L-16.pth")
  parser.add_argument("--model", default="JiT-L/16", choices=["JiT-L/16", "JiT-B/16"])
  parser.add_argument("--batch_size", default=16)
  parser.add_argument("--save_last_freq", default=700)
  parser.add_argument("--eval_freq", default=100)
  parser.add_argument("--sampling_method", default="heun")
  parser.add_argument("--num_sampling_steps", default=50)
  args, passthru = parser.parse_known_args()

  # Step 1: Generate glyph dataset
  generate_glyph_dataset(
    args.source_font,
    args.glyph_dir,
    args.train_count,
    output_dir=args.dataset_output_dir,
  )

  # Step 2: Fine-tune the model
  finetune_model(passthru, 
    dataset_dir=args.dataset_output_dir, 
    output_dir=args.output_dir,
    epochs=args.epochs,
    base_checkpoint=args.base_checkpoint,
    model=args.model,
    batch_size=args.batch_size,
    save_last_freq=args.save_last_freq,
    eval_freq=args.eval_freq,
    sampling_method=args.sampling_method,
    num_sampling_steps=args.num_sampling_steps,
  )


if __name__ == "__main__":
  main()
