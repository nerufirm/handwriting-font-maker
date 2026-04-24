---
title: Handwriting Font Maker
emoji: 🖋️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🖋️ 手書き文字フォント作成ツール

手書きの文字画像をアップロードするだけで、自分だけのTTFフォントを生成するStreamlitアプリです。

## 使い方

1. 118文字（ひらがな・カタカナ・漢字・英数字・記号）を順番通りに手書きした画像を用意
2. アプリにアップロード
3. 「フォントを生成する」をクリック
4. 完成したTTFをダウンロード

## 仕組み

- **OpenCV** — 文字を1文字ずつ自動切り出し
- **potrace** — ビットマップをSVGベクターに変換
- **FontForge** — SVGをまとめてTTFフォントに合成

## ローカル実行

```bash
pip install -r requirements.txt
# システムにpotrace, fontforgeも必要 (macOS: brew install potrace fontforge)
streamlit run app.py
```
