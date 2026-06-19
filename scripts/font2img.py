import os
from font2data.FontData import FontData
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--common_chars", type=str, default="./scripts/Chinese_characters_3500.txt")
parser.add_argument("--font_file", type=str, required=True)
parser.add_argument("--output_folder", type=str, default="./data/sample_glyphs")

if __name__ == "__main__":
    
    args = parser.parse_args()


    with open(args.common_chars, "r", encoding="utf-8") as f:
        text = f.read()
        # 去掉空白與換行，再拆成單個字
        common_chars = list(text.replace(" ", "").replace("\n", "").strip())
        # 如需去重，可保留下一行，否則刪掉
        common_chars = list(set(common_chars))
        print("len(common_chars): ", len(common_chars))

    #clear output folder``
    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)
    os.makedirs(args.output_folder, exist_ok=True)

    font_data = FontData(f"{args.font_file}", font_size=256)
    for char in common_chars:
        image = font_data.char2img(char)
        if image is not None:
            # image = image.convert("L")
            # 使用 Unicode 碼點 (hex) 當檔名，避免 Windows 中文檔名編碼問題
            # name = f"{ord(char):04X}.png"
            name = f"{char}.png"
            image.save(os.path.join(args.output_folder, name))
