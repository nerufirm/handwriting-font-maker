import os
import subprocess
import cv2
from pathlib import Path

def main():
    input_dir = Path("output")
    svg_dir = Path("svg_chars")
    svg_dir.mkdir(exist_ok=True)

    print("1/2: 画像をベクターデータ(SVG)に変換しています...")
    
    for img_path in input_dir.glob("*.png"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        bmp_path = svg_dir / f"{img_path.stem}.bmp"
        svg_path = svg_dir / f"{img_path.stem}.svg"
        cv2.imwrite(str(bmp_path), img)

        subprocess.run(["potrace", str(bmp_path), "-s", "-o", str(svg_path)], check=True)
        bmp_path.unlink()

    print("SVGの生成が完了しました。")
    print("2/2: FontForgeでTTFフォントをコンパイルしています...")

    ff_script = """import fontforge
import os

font = fontforge.font()
font.fontname = "KswSyouwa"
font.familyname = "Ksw Syouwa"
font.fullname = "Ksw Syouwa Regular"
font.encoding = "UnicodeFull"

svg_dir = "svg_chars"

def get_unicode(filename):
    name = filename.replace(".svg", "")
    if name.startswith("upper_"): return ord(name[6:])
    if name.startswith("lower_"): return ord(name[6:])
    
    sym_map = {
        "dot": ".", "comma": ",", "colon": ":", "semicolon": ";",
        "exclamation": "!", "question": "?", "single_quote": "'",
        "double_quote": '"', "dollar": "$", "percent": "%",
        "ampersand": "&", "at": "@"
    }
    if name in sym_map:
        return ord(sym_map[name])
        
    if len(name) == 1: return ord(name)
    return None

for f in os.listdir(svg_dir):
    if not f.endswith(".svg"): continue
    uni = get_unicode(f)
    if uni:
        char = font.createChar(uni)
        char.importOutlines(os.path.join(svg_dir, f))
        
        # 修正箇所: 文字の幅を「図形の右端座標 + 50(余白)」に設定
        bbox = char.boundingBox()
        char.width = int(bbox[2] + 50)

font.generate("KswSyouwa.ttf")
"""

    with open("ff_generate.py", "w", encoding="utf-8") as f:
        f.write(ff_script)

    subprocess.run(["fontforge", "-lang=py", "-script", "ff_generate.py"], check=True)
    os.remove("ff_generate.py")
    
    print("🎉 すべての工程が完了しました！ 作業フォルダに 'KswSyouwa.ttf' が生成されています。")

if __name__ == "__main__":
    main()