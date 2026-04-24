"""
手書き文字画像から1文字ずつ画像を切り出して個別ファイルに保存するスクリプト。

処理フロー:
    1. グレースケール化
    2. 二値化（Otsu法）
    3. 膨張処理: 縦長カーネルで「i/jの点」「濁点・半濁点」「漢字の偏旁」を結合
    4. 輪郭抽出 → 面積フィルタでノイズ除去
    5. Y座標で行にグループ化 → 各行をX座標で左→右にソート
    6. 行内の後処理:
         - 幅広ボックスの縦投影分割（隣接漢字などの誤結合を解消）
         - 狭い隣接ボックスの結合（「い」など2本線の字の救済）
    7. 膨張前の綺麗な二値化画像から、パディング付きで切り出して保存

使い方:
    # 依存関係をインストール
    pip install opencv-python numpy

    # スクリプトと同じ階層に入力画像を置いて実行
    python extract_chars.py
"""

import os
from pathlib import Path

import cv2
import numpy as np

# ===== 設定（画像特性に合わせて調整） =========================
INPUT_IMAGE = "ksw_syouwa.webp"
OUTPUT_DIR = "output"

# 膨張カーネル: 縦を長めにして「点」「濁点」「偏旁」を本体と結合。
# 横は控えめにして隣の文字と過度に結合しないようにする
DILATE_KERNEL_SIZE = (5, 15)     # (幅, 高さ)
DILATE_ITERATIONS = 2

# ノイズ除去: バウンディングボックスの最小面積
MIN_AREA = 150

# 行グループ化: 中央値高さ × この係数 以内なら同じ行
ROW_THRESHOLD_RATIO = 0.6

# 行内の後処理（狭い隣接ボックスの結合）
#   NARROW_RATIO: 行中央値幅 × この係数より狭いボックスは「狭い」とみなす
#   TINY_RATIO:   これ以下なら単独でも隣と結合する
#   MAX_GAP_RATIO: 横方向ギャップが 行中央値幅 × この係数以内なら結合対象
NARROW_RATIO = 0.55
TINY_RATIO = 0.40
MAX_GAP_RATIO = 0.30

# 行内の後処理（幅広ボックスの分割）
#   WIDE_RATIO: 行中央値幅 × この係数を超えるボックスは「2文字以上結合」候補
#   SPLIT_VALLEY_RATIO: 谷の深さがピークの何割以下なら分割点とみなすか
WIDE_RATIO = 1.6
SPLIT_VALLEY_RATIO = 0.15

# 保存時のパディング（px）
PADDING = 10
# =============================================================


def _group_into_rows(
    bboxes: list[tuple[int, int, int, int]],
    median_h: float,
) -> list[list[tuple[int, int, int, int]]]:
    """バウンディングボックスをY座標で行にグループ化する。

    Y中心座標でソートし、行の平均Y中心から `median_h * ROW_THRESHOLD_RATIO`
    以内なら同じ行として扱う（ランニング平均で行の重心を更新）。
    """
    row_threshold = median_h * ROW_THRESHOLD_RATIO
    bboxes_with_yc = sorted(
        [(x, y, w, h, y + h / 2.0) for (x, y, w, h) in bboxes],
        key=lambda b: b[4],
    )

    rows: list[list[tuple[int, int, int, int]]] = []
    current_row: list[tuple[int, int, int, int]] = []
    current_row_yc: float | None = None

    for x, y, w, h, yc in bboxes_with_yc:
        if current_row_yc is None or abs(yc - current_row_yc) < row_threshold:
            current_row.append((x, y, w, h))
            n = len(current_row)
            current_row_yc = (
                yc if n == 1 else (current_row_yc * (n - 1) + yc) / n
            )
        else:
            rows.append(current_row)
            current_row = [(x, y, w, h)]
            current_row_yc = yc
    if current_row:
        rows.append(current_row)
    return rows


def _split_wide_box(
    bbox: tuple[int, int, int, int],
    binary: np.ndarray,
    median_w: float,
) -> list[tuple[int, int, int, int]]:
    """幅が中央値より大きすぎるボックスを縦投影から分割する。

    元の二値化画像（膨張前）に対して、ボックス内の各列の白画素数を
    数え、谷（画素数がピークの `SPLIT_VALLEY_RATIO` 倍以下の連続領域）
    をすべて検出する。谷の中心位置で分割し、分割後の各幅が
    「中央値 × NARROW_RATIO」以上になるものだけ採用する。
    """
    x, y, w, h = bbox
    if w <= median_w * WIDE_RATIO:
        return [bbox]

    region = binary[y : y + h, x : x + w]
    col_sum = np.sum(region > 0, axis=0)  # 各列の前景画素数
    peak = float(col_sum.max()) if col_sum.size else 0.0
    if peak <= 0:
        return [bbox]
    valley_threshold = peak * SPLIT_VALLEY_RATIO

    # 谷領域（連続する低画素列）を検出
    is_valley = col_sum <= valley_threshold
    valleys: list[tuple[int, int]] = []
    i = 0
    W = len(col_sum)
    while i < W:
        if is_valley[i]:
            j = i
            while j < W and is_valley[j]:
                j += 1
            # 端の谷（余白）は除外し、内部の谷だけ分割候補に
            if i > 0 and j < W:
                valleys.append((i, j))
            i = j
        else:
            i += 1

    if not valleys:
        return [bbox]

    # 各谷の中心でボックスを分割
    cut_positions = [(a + b) // 2 for (a, b) in valleys]
    sub_boxes: list[tuple[int, int, int, int]] = []
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

    # 有効な分割が得られなかった場合は元のボックスを返す
    if len(sub_boxes) < 2:
        return [bbox]
    return sub_boxes


def _refine_row(
    row: list[tuple[int, int, int, int]],
    binary: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """行内で極端に狭い隣接ボックスを結合する。

    「い」のように縦2本線で構成される字は、縦優位の膨張だけでは
    結合されないことがある。行の中央値幅を基準に、不自然に狭い
    ボックス同士（または片方が極小）を近接距離で結合する。
    """
    row = sorted(row, key=lambda b: b[0])
    if not row:
        return row

    widths = [w for (_, _, w, _) in row]
    median_w = float(np.median(widths))

    # --- (a) 幅広ボックスの分割（複数文字が結合してしまったケースの救済）---
    expanded: list[tuple[int, int, int, int]] = []
    for bbox in row:
        expanded.extend(_split_wide_box(bbox, binary, median_w))
    row = sorted(expanded, key=lambda b: b[0])
    if len(row) < 2:
        return row

    # 分割後の中央値で閾値を再計算
    widths = [w for (_, _, w, _) in row]
    median_w = float(np.median(widths))
    narrow_threshold = median_w * NARROW_RATIO
    tiny_threshold = median_w * TINY_RATIO
    max_gap = median_w * MAX_GAP_RATIO

    # --- (b) 狭い隣接ボックスの結合（分離しやすい字の救済）---
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
                # 2つのボックスを包含する新しいボックス
                nx = min(x1, x2)
                ny = min(y1, y2)
                nx2 = max(x1 + w1, x2 + w2)
                ny2 = max(y1 + h1, y2 + h2)
                row[i] = (nx, ny, nx2 - nx, ny2 - ny)
                row.pop(i + 1)
                changed = True
                break
    return row


def extract_characters(image_path: str, output_dir: str) -> int:
    # --- 1. 画像読み込み ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めませんでした: {image_path}")

    # --- 2. グレースケール化 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 3. 二値化（Otsu法 + 反転: 文字=255, 背景=0） ---
    # 輪郭抽出は前景を白として扱うため反転する
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --- 4. 膨張処理（結合処理） ---
    # 縦長カーネルで i/j の点、濁点・半濁点、漢字の偏旁などを1つの塊に
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KERNEL_SIZE)
    dilated = cv2.dilate(binary, kernel, iterations=DILATE_ITERATIONS)

    # --- 5. 輪郭抽出（外側の輪郭のみ） + 面積フィルタ ---
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= MIN_AREA:
            bboxes.append((x, y, w, h))

    if not bboxes:
        print("文字が検出されませんでした。パラメータを調整してください。")
        return 0

    # --- 6. 行でグループ化 → 各行を左→右にソート → 行内後処理 ---
    median_h = float(np.median([h for (_, _, _, h) in bboxes]))
    rows = _group_into_rows(bboxes, median_h)
    rows = [_refine_row(r, binary) for r in rows]

    sorted_bboxes: list[tuple[int, int, int, int]] = []
    for row in rows:
        sorted_bboxes.extend(sorted(row, key=lambda b: b[0]))

    # --- 7. 出力フォルダ作成 ---
    os.makedirs(output_dir, exist_ok=True)

    # --- 8. 切り出して保存 ---
    # 保存は「膨張前の綺麗な二値化画像」を白背景・黒文字にして使う
    clean_binary = cv2.bitwise_not(binary)
    H, W = clean_binary.shape

    for i, (x, y, w, h) in enumerate(sorted_bboxes, start=1):
        x0 = max(0, x - PADDING)
        y0 = max(0, y - PADDING)
        x1 = min(W, x + w + PADDING)
        y1 = min(H, y + h + PADDING)
        char_img = clean_binary[y0:y1, x0:x1]
        out_path = os.path.join(output_dir, f"char_{i:03d}.png")
        cv2.imwrite(out_path, char_img)

    return len(sorted_bboxes)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / INPUT_IMAGE
    output_path = base_dir / OUTPUT_DIR

    count = extract_characters(str(input_path), str(output_path))
    print(f"{count} 個の文字を {output_path}/ に保存しました")
