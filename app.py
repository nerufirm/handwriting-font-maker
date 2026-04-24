import streamlit as st
import cv2
import numpy as np
import os
import subprocess
from pathlib import Path
import tempfile
import shutil
from PIL import Image

# --- 定数設定 ---
CHARS = "あいうえおかきくけこさしすせそアイウエオカキクケコサシスセソ安以宇衣於加幾久計己左之寸世曽ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz123456789&.,;:!?'$\"%@"
SYMBOL_MAP = {
    ".": "dot", ",": "comma", ":": "colon", ";": "semicolon",
    "!": "excl", "?": "ques", "'": "sq", "\"": "dq",
    "$": "dollar", "%": "per", "&": "amp", "@": "at"
}

# --- 画像処理ロジック (ローカル版 extract_chars.py を移植) ---
def group_into_rows(bboxes, median_h):
    bboxes_yc = sorted([(x, y, w, h, y + h / 2) for (x, y, w, h) in bboxes], key=lambda b: b[4])
    rows, cur, cur_yc = [], [], None
    for x, y, w, h, yc in bboxes_yc:
        if cur_yc is None or abs(yc - cur_yc) < median_h * 0.7:
            cur.append((x, y, w, h))
            n = len(cur)
            cur_yc = yc if n == 1 else (cur_yc * (n - 1) + yc) / n
        else:
            rows.append(cur)
            cur, cur_yc = [(x, y, w, h)], yc
    if cur:
        rows.append(cur)
    return rows

def refine_row(row, binary_img, med_h):
    row = sorted(row, key=lambda b: b[0])
    changed = True
    while changed:
        changed = False
        for i in range(len(row) - 1):
            x1, y1, w1, h1 = row[i]
            x2, y2, w2, h2 = row[i + 1]
            if (x2 - (x1 + w1)) < (med_h * 0.3):
                nx, ny = min(x1, x2), min(y1, y2)
                row[i] = (nx, ny, max(x1 + w1, x2 + w2) - nx, max(y1 + h1, y2 + h2) - ny)
                row.pop(i + 1)
                changed = True
                break
    return row

# --- Streamlit UI ---
st.set_page_config(page_title="Handwriting Font Maker", layout="wide")
st.title("🖋️ 手書き文字フォント作成ツール")
st.write("画像をアップロードするだけで、自分だけのフォント(.ttf)を作成します。")

uploaded_file = st.file_uploader(
    "手書き文字の画像をアップロード (118文字並んでいるもの)",
    type=['png', 'jpg', 'jpeg', 'webp'],
)

if not uploaded_file:
    st.info(
        "👋 **まずはじめに、元となる手書き文字の画像ファイルをアップロードしてください。**\n\n"
        "上の枠に画像をドラッグ＆ドロップするか、ファイルを選択するとフォント生成の準備ができます。"
    )
    with st.expander("📝 画像の準備について"):
        st.markdown(
            f"""
- 手書きで **{len(CHARS)}文字** を順番通りに書いた画像を用意してください。
- 文字の順序（左上から右下へ、行ごとに左から右）：
  ```
  {CHARS}
  ```
- 推奨フォーマット: PNG / JPG / WebP（背景は白、文字は黒推奨）
- 文字どうしが重ならないよう、適度な間隔を空けて書いてください。
            """
        )

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        output_dir = tmp_path / "output"
        svg_dir = tmp_path / "svg"
        output_dir.mkdir()
        svg_dir.mkdir()

        # 画像読み込み
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="アップロードされた画像", use_container_width=True)

        if st.button("フォントを生成する"):
            with st.spinner("文字を切り出してフォントを構築中..."):
                # 1. 切り出し
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
                dilated = cv2.dilate(binary, kernel, iterations=2)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
                if len(bboxes) < 50:
                    st.error("文字が十分に検出されませんでした。")
                else:
                    med_h = np.median([h for (_, _, _, h) in bboxes])
                    rows = group_into_rows(bboxes, med_h)
                    rows = [refine_row(r, binary, med_h) for r in rows]

                    flat_bboxes = []
                    for r in rows:
                        flat_bboxes.extend(sorted(r, key=lambda b: b[0]))

                    # 2. SVG化 & 3. FontForge
                    # (リネームとSVG生成を同時に行う)
                    for i, (x, y, w, h) in enumerate(flat_bboxes):
                        if i >= len(CHARS):
                            break
                        char_img = binary[y:y + h, x:x + w]
                        char_img = cv2.copyMakeBorder(char_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)

                        bmp_p = tmp_path / f"temp_{i}.bmp"
                        svg_p = svg_dir / f"char_{i}.svg"
                        cv2.imwrite(str(bmp_p), char_img)
                        subprocess.run(["potrace", str(bmp_p), "-s", "-o", str(svg_p)])

                    # FontForge スクリプト作成
                    # repr() で文字列をエスケープし、CHARS内の " や特殊文字で構文を壊さないようにする
                    chars_repr = repr(CHARS)
                    svg_dir_repr = repr(str(svg_dir))
                    out_ttf_repr = repr(str(tmp_path / "font.ttf"))
                    ff_script = f"""
import fontforge
import os

font = fontforge.font()
font.fontname = "MyHandFont"
font.familyname = "My Hand Font"
font.fullname = "My Hand Font Regular"
chars_list = {chars_repr}
svg_dir = {svg_dir_repr}
for i in range(len(chars_list)):
    svg_file = os.path.join(svg_dir, f"char_{{i}}.svg")
    if os.path.exists(svg_file):
        char = font.createChar(ord(chars_list[i]))
        char.importOutlines(svg_file)
        bbox = char.boundingBox()
        char.width = int(bbox[2] + 50)
font.generate({out_ttf_repr})
"""
                    with open(tmp_path / "gen.py", "w") as f:
                        f.write(ff_script)
                    subprocess.run(["fontforge", "-lang=py", "-script", str(tmp_path / "gen.py")])

                    # ダウンロード
                    font_file = tmp_path / "font.ttf"
                    if font_file.exists():
                        st.success("フォントが完成しました！")
                        with open(font_file, "rb") as f:
                            st.download_button("TTFファイルをダウンロード", f, file_name="HandwritingFont.ttf")
                    else:
                        st.error("フォント生成に失敗しました。")
