"""
`output/` フォルダ内の `char_001.png` 〜 `char_118.png` を、
抽出済みの読み順（左上→右下）に基づいて実際の文字名にリネームする。

使い方:
    python rename_chars.py

想定:
    - `extract_chars.py` が正しく 118 枚を生成済みであること
    - 連番ファイルの順番が、下記 CHARS の順番と 1 対 1 対応すること
"""

from __future__ import annotations

import sys
from pathlib import Path

# ===== リネーム対象の文字列（読み順、計118文字） =====
CHARS = (
    "あいうえおかきくけこさしすせそ"
    "アイウエオカキクケコサシスセソ"
    "安以宇衣於加幾久計己左之寸世曽"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "123456789&.,;:!?'$\"%@"
)

# OSでファイル名に使えない/不都合な記号の置換名
# 一般的に避けたい記号: / \ : * ? " < > |（Windows）/（Unix）
SPECIAL_NAMES: dict[str, str] = {
    ".": "dot",
    ",": "comma",
    ":": "colon",
    ";": "semicolon",
    "!": "exclamation",
    "?": "question",
    "'": "single_quote",
    '"': "double_quote",
    "$": "dollar",
    "%": "percent",
    "&": "ampersand",
    "@": "at",
}

# 設定
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
EXPECTED_COUNT = 118


def char_to_filename_stem(char: str) -> str:
    """文字から安全なファイル名（拡張子なしのステム）を生成する。

    - 特殊記号は `SPECIAL_NAMES` で置換
    - ASCII 英字は大文字小文字を区別しないファイルシステム（macOS/Windows既定）で
      衝突しないよう `upper_A` / `lower_a` のプレフィックス形式にする
    - ひらがな/カタカナ/漢字/数字はそのまま使う
    """
    if char in SPECIAL_NAMES:
        return SPECIAL_NAMES[char]
    if "A" <= char <= "Z":
        return f"upper_{char}"
    if "a" <= char <= "z":
        return f"lower_{char}"
    return char


def main() -> int:
    if not OUTPUT_DIR.is_dir():
        print(f"ERROR: 出力フォルダが見つかりません: {OUTPUT_DIR}", file=sys.stderr)
        return 1

    # `char_NNN.png` を番号順に収集
    src_files = sorted(OUTPUT_DIR.glob("char_*.png"))
    if len(src_files) != EXPECTED_COUNT:
        print(
            f"ERROR: {EXPECTED_COUNT} 枚の char_*.png が必要ですが、"
            f"{len(src_files)} 枚しか見つかりませんでした。",
            file=sys.stderr,
        )
        return 1
    if len(CHARS) != EXPECTED_COUNT:
        print(
            f"ERROR: CHARS の長さが {len(CHARS)} で {EXPECTED_COUNT} と一致しません。",
            file=sys.stderr,
        )
        return 1

    # 1対1にマッピング。重複検知のため事前検証
    target_stems = [char_to_filename_stem(c) for c in CHARS]
    if len(set(target_stems)) != len(target_stems):
        # 重複があれば、どの文字かを提示して停止
        from collections import Counter

        counter = Counter(target_stems)
        dups = {k: v for k, v in counter.items() if v > 1}
        print(f"ERROR: リネーム先が重複しています: {dups}", file=sys.stderr)
        return 1

    # 2段階リネーム（案）は不要。プレフィックス/置換により
    # `char_NNN.png` とは名前空間が衝突しないため直接 rename できる。
    renamed = 0
    for src, char, stem in zip(src_files, CHARS, target_stems):
        dst = OUTPUT_DIR / f"{stem}.png"
        if dst.exists() and dst.resolve() != src.resolve():
            print(
                f"WARNING: 既存ファイルを上書きします: {dst.name}",
                file=sys.stderr,
            )
            dst.unlink()
        src.rename(dst)
        renamed += 1
        print(f"  {src.name} -> {dst.name}  ({char!r})")

    print(f"\n{renamed} 個のファイルをリネームしました → {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
