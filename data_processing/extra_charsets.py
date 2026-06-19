import os
from font2data import words
from . import charsets

def get_charset_codepoints(charset: str | list[str]) -> set[int]:

    charset_cps = process_chars(charset, return_unicode=True)
    return set(charset_cps)

    # if charset == 'gbk':
    #     # CJK统一表意文字
    #     charset_cps = set(range(0x4E00, 0x9FFF + 1))
    #     print(f"Using hardcoded GBK range: {len(charset_cps)} codepoints")
    # else:
    #     # 默认GB2312
    #     charset_cps = charsets.get_charset_codepoints(charset)
    #     print(f"Charset {charset}: {len(charset_cps)} codepoints")

    # return charset_cps


def read_chars_from_file(file_path):
    """從文件中讀取字符"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到字符文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        return list(set(content))

def process_chars(input_chars: str | list[str], return_unicode: bool = False):
    """處理字符輸入，支援字串、列表或檔案路徑"""
    generate_chars = []
    
    # Case 1: 字串輸入（例如 default="generate_chars.txt" 或直接傳 "A,B,C"）
    if isinstance(input_chars, str):
        # Case 1a: 字串為 .txt 檔案路徑 → 讀取檔案內所有字元
        if input_chars.endswith(".txt"):
            generate_chars = read_chars_from_file(input_chars)
        # Case 1b: 逗號分隔字串 → "A,B,C,我"
        else:
            generate_chars = input_chars.split(",")
    
    # Case 2: 列表輸入（CLI --generate_chars 使用 nargs='+'）
    elif isinstance(input_chars, list):
        items = input_chars
        # Case 2a: 單一引號參數且含空格 → CLI: --generate_chars "A B C 我" → ['A B C 我']
        # 排除 .txt 路徑與 wordset 名稱，避免誤拆
        if (len(items) == 1 and isinstance(items[0], str)
                and " " in items[0].strip()
                and not items[0].endswith(".txt")
                and items[0] not in words.words_dict.keys()):
            items = items[0].split()
        # Case 2b~2f: 逐項處理
        #   2b: 多參數 CLI → --generate_chars A B C 我 → ['A','B','C','我']
        #   2f: 混合列表 → ['A','B','C','我','data/chars.txt']（字元與 .txt 可混用）
        for item in items:
            # Case 2c: 列表項為 .txt 檔案路徑 → ['data/chars.txt', ...]
            if isinstance(item, str) and item.endswith(".txt"):
                generate_chars.extend(read_chars_from_file(item))
            # Case 2d: 列表項為 wordset 名稱 → 展開為該字集所有字元
            elif isinstance(item, str) and item in words.words_dict.keys():
                words_dict = words.get_wordset(item)
                for k, v in words_dict.items():
                    generate_chars.extend([chr(unicode) for unicode in v])
            # Case 2e: 單一字元 → ['A', 'B', '我', ...]
            else:
                generate_chars.append(item)

    if return_unicode:
        generate_chars = [ord(ch) for ch in generate_chars]
    
    return list(set(generate_chars))