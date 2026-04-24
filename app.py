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

# 画像処理パラメータ（extract_chars.py の実証済み値）
DILATE_KERNEL_SIZE = (5, 15)
DILATE_ITERATIONS = 2
MIN_AREA = 150
ROW_THRESHOLD_RATIO = 0.6
NARROW_RATIO = 0.55
TINY_RATIO = 0.40
MAX_GAP_RATIO = 0.30
WIDE_RATIO = 1.6
SPLIT_VALLEY_RATIO = 0.15
PADDING = 10


# --- 画像処理ロジック (extract_chars.py からの移植) ---
def group_into_rows(bboxes, median_h):
    """Y座標で行にグループ化"""
    row_threshold = median_h * ROW_THRESHOLD_RATIO
    bboxes_yc = sorted(
        [(x, y, w, h, y + h / 2.0) for (x, y, w, h) in bboxes],
        key=lambda b: b[4],
    )
    rows = []
    cur, cur_yc = [], None
    for x, y, w, h, yc in bboxes_yc:
        if cur_yc is None or abs(yc - cur_yc) < row_threshold:
            cur.append((x, y, w, h))
            n = len(cur)
            cur_yc = yc if n == 1 else (cur_yc * (n - 1) + yc) / n
        else:
            rows.append(cur)
            cur, cur_yc = [(x, y, w, h)], yc
    if cur:
        rows.append(cur)
    return rows


def split_wide_box(bbox, binary, median_w):
    """幅広ボックスを縦投影の谷で分割（複数文字が結合してしまったケースの救済）"""
    x, y, w, h = bbox
    if w <= median_w * WIDE_RATIO:
        return [bbox]

    region = binary[y:y + h, x:x + w]
    col_sum = np.sum(region > 0, axis=0)
    peak = float(col_sum.max()) if col_sum.size else 0.0
    if peak <= 0:
        return [bbox]
    valley_threshold = peak * SPLIT_VALLEY_RATIO

    is_valley = col_sum <= valley_threshold
    valleys = []
    i = 0
    W = len(col_sum)
    while i < W:
        if is_valley[i]:
            j = i
            while j < W and is_valley[j]:
                j += 1
            if i > 0 and j < W:
                valleys.append((i, j))
            i = j
        else:
            i += 1

    if not valleys:
        return [bbox]

    cut_positions = [(a + b) // 2 for (a, b) in valleys]
    sub_boxes = []
    prev = 0
    min_sub_w = median_w * NARROW_RATIO
    for cx in cut_positions:
        sub_w = cx - prev
        if sub_w >= min_sub_w:
            sub_boxes.append((x + prev, y, sub_w, h))
        prev = cx
    tail_w = w - prev
    if tail_w >= min_sub_w:
        sub_boxes.append((x + prev, y, tail_w, h))

    if len(sub_boxes) < 2:
        return [bbox]
    return sub_boxes


def refine_row(row, binary_img):
    """行内ボックスの後処理: 幅広分割 + 狭い隣接結合"""
    row = sorted(row, key=lambda b: b[0])
    if not row:
        return row

    widths = [w for (_, _, w, _) in row]
    median_w = float(np.median(widths))

    # (a) 幅広ボックスの分割
    expanded = []
    for bbox in row:
        expanded.extend(split_wide_box(bbox, binary_img, median_w))
    row = sorted(expanded, key=lambda b: b[0])
    if len(row) < 2:
        return row

    # 分割後の中央値で再計算
    widths = [w for (_, _, w, _) in row]
    median_w = float(np.median(widths))
    narrow_threshold = median_w * NARROW_RATIO
    tiny_threshold = median_w * TINY_RATIO
    max_gap = median_w * MAX_GAP_RATIO

    # (b) 狭い隣接ボックスのみ結合（「い」など2本線字の救済）
    changed = True
    while changed:
        changed = False
        for i in range(len(row) - 1):
            x1, y1, w1, h1 = row[i]
            x2, y2, w2, h2 = row[i + 1]
            gap = x2 - (x1 + w1)
            both_narrow = w1 < narrow_threshold and w2 < narrow_threshold
            one_tiny = min(w1, w2) < tiny_threshold
            if gap < max_gap and (both_narrow or one_tiny):
                nx = min(x1, x2)
                ny = min(y1, y2)
                nx2 = max(x1 + w1, x2 + w2)
                ny2 = max(y1 + h1, y2 + h2)
                row[i] = (nx, ny, nx2 - nx, ny2 - ny)
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
        svg_dir = tmp_path / "svg"
        svg_dir.mkdir()

        # 画像読み込み
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="アップロードされた画像", use_container_width=True)

        if st.button("フォントを生成する"):
            with st.spinner("文字を切り出してフォントを構築中..."):
                # 1. 切り出し（extract_chars.py のロジックに準拠）
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KERNEL_SIZE)
                dilated = cv2.dilate(binary, kernel, iterations=DILATE_ITERATIONS)
                contours, _ = cv2.findContours(
                    dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                bboxes = []
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    if w * h >= MIN_AREA:
                        bboxes.append((x, y, w, h))

                if len(bboxes) < 50:
                    st.error(f"文字が十分に検出されませんでした（{len(bboxes)}個）。画像の品質を確認してください。")
                else:
                    median_h = float(np.median([h for (_, _, _, h) in bboxes]))
                    rows = group_into_rows(bboxes, median_h)
                    rows = [refine_row(r, binary) for r in rows]

                    # 行ごとのベースラインを推定（文字下端の中央値 ≒ ベースライン）
                    row_baselines = []
                    flat_bboxes_with_row = []
                    for r_idx, r in enumerate(rows):
                        r_sorted = sorted(r, key=lambda b: b[0])
                        if not r_sorted:
                            row_baselines.append(0)
                            continue
                        bottoms = [by + bh for (_, by, _, bh) in r_sorted]
                        row_baselines.append(int(np.median(bottoms)))
                        for bbox in r_sorted:
                            flat_bboxes_with_row.append(bbox + (r_idx,))

                    flat_bboxes = [b[:4] for b in flat_bboxes_with_row]
                    st.write(f"検出文字数: {len(flat_bboxes)} / 期待値: {len(CHARS)}")

                    # 全文字に共通の ascent/descent を算出（行ベースライン基準）
                    max_ascent_px = 0
                    max_descent_px = 0
                    for (bx, by, bw, bh, r_idx) in flat_bboxes_with_row:
                        baseline = row_baselines[r_idx]
                        max_ascent_px = max(max_ascent_px, baseline - by)
                        max_descent_px = max(max_descent_px, (by + bh) - baseline)

                    # 安全マージン
                    safety = int(max(max_ascent_px, max_descent_px, 1) * 0.15)
                    canvas_ascent_px = max_ascent_px + safety
                    canvas_descent_px = max_descent_px + safety
                    canvas_h = canvas_ascent_px + canvas_descent_px

                    # キャンバス幅は全文字最大幅にマージン（全SVGで同一）
                    max_w_px = max(bw for (_, _, bw, _) in flat_bboxes)
                    canvas_w = int(max_w_px * 1.5)

                    # 2. 各文字を統一サイズのキャンバス上に配置してSVG化
                    # potraceは「黒=前景」として読むため、bitwise_notで白背景・黒文字に反転
                    clean_binary = cv2.bitwise_not(binary)  # 黒文字 / 白背景
                    H, W = clean_binary.shape

                    for i, (x, y, w, h, r_idx) in enumerate(flat_bboxes_with_row):
                        if i >= len(CHARS):
                            break
                        baseline = row_baselines[r_idx]

                        # 元画像から文字インク部分を切り出す（パディングなし・タイト）
                        x0 = max(0, x)
                        y0 = max(0, y)
                        x1 = min(W, x + w)
                        y1 = min(H, y + h)
                        char_img = clean_binary[y0:y1, x0:x1]
                        ch_h, ch_w = char_img.shape[:2]

                        # 統一キャンバス（白背景）を作成
                        canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

                        # ベースラインを canvas_ascent_px に置き、文字を配置
                        top_in_canvas = canvas_ascent_px - (baseline - y)
                        left_in_canvas = (canvas_w - ch_w) // 2

                        # 境界クリップ
                        top_in_canvas = max(0, min(canvas_h - ch_h, top_in_canvas))
                        left_in_canvas = max(0, min(canvas_w - ch_w, left_in_canvas))

                        canvas[top_in_canvas:top_in_canvas + ch_h,
                               left_in_canvas:left_in_canvas + ch_w] = char_img

                        bmp_p = tmp_path / f"temp_{i}.bmp"
                        svg_p = svg_dir / f"char_{i}.svg"
                        cv2.imwrite(str(bmp_p), canvas)
                        subprocess.run(
                            ["potrace", str(bmp_p), "-s", "-o", str(svg_p)],
                            check=False,
                        )

                    # 3. FontForge スクリプトでTTF合成
                    # キャンバス上の baseline 位置（上端から）に相当する em 単位の値
                    # SVG は canvas_h pt 高さ。FontForge は pt→em を線形にスケール。
                    # em=1000 想定で ascent/descent を設定。
                    EM = 1000
                    # canvas_h を em に対応付ける。ascent:descent = canvas_ascent_px:canvas_descent_px
                    em_ascent = int(EM * canvas_ascent_px / canvas_h)
                    em_descent = EM - em_ascent

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
font.encoding = "UnicodeFull"
font.em = {EM}
font.ascent = {em_ascent}
font.descent = {em_descent}

# .notdef を空グリフにして、未定義文字がベタ黒ブロックにならないようにする
notdef = font.createChar(-1, ".notdef")
notdef.width = int({EM} * 0.5)

chars_list = {chars_repr}
svg_dir = {svg_dir_repr}
side_bearing = int({EM} * 0.05)
advance_widths = []
for i in range(len(chars_list)):
    svg_file = os.path.join(svg_dir, f"char_{{i}}.svg")
    if not os.path.exists(svg_file):
        continue
    char = font.createChar(ord(chars_list[i]))
    char.importOutlines(svg_file)
    bb = char.boundingBox()  # (xmin, ymin, xmax, ymax)
    # advance width = インク右端 + side_bearing（左側の余白は自然なまま）
    char.width = int(bb[2] + side_bearing)
    advance_widths.append(char.width)

# 半角スペース
space = font.createChar(ord(" "), "space")
if advance_widths:
    space.width = int(sum(advance_widths) / len(advance_widths) * 0.5)
else:
    space.width = int({EM} * 0.3)

font.generate({out_ttf_repr})
"""
                    with open(tmp_path / "gen.py", "w") as f:
                        f.write(ff_script)
                    subprocess.run(
                        ["fontforge", "-lang=py", "-script", str(tmp_path / "gen.py")],
                        check=False,
                    )

                    # 4. ダウンロード
                    font_file = tmp_path / "font.ttf"
                    if font_file.exists():
                        st.success("フォントが完成しました！")
                        with open(font_file, "rb") as f:
                            st.download_button(
                                "TTFファイルをダウンロード",
                                f,
                                file_name="HandwritingFont.ttf",
                            )
                    else:
                        st.error("フォント生成に失敗しました。")
