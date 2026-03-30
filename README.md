# Qwen3.5 + SGLang GPU Demo

Qwen3.5 を **SGLang** で起動し、**GPU を使って** テキスト入力と**画像入力（vision input）**の両方を試すデモです。

このリポジトリでは次を提供します。
- Qwen3.5-4B を SGLang サーバとして起動するスクリプト
- OpenAI互換 API でテキスト/画像入力を送るサンプル
- 動作確認用のテスト画像生成スクリプト
- 同内容を Jupyter Notebook で再実行できる `.ipynb`
- 環境構築手順
- Dockerfile

## 重要な前提

- NVIDIA GPU が使えること
- Docker が使えること
- Hugging Face からモデル取得できること
- 必要に応じて `HF_TOKEN` を設定すること

> このプロジェクトでは、ホスト側で SGLang を直接ビルドする代わりに、**公式 SGLang CUDA 13 runtime Docker image** を使います。今回の arm64 + CUDA 13 環境では、この方法が最も再現しやすいためです。

## 採用モデル

- 既定モデル: `Qwen/Qwen3.5-4B`

理由:
- Hugging Face モデルカード上で **"Causal Language Model with Vision Encoder"** と明記されている
- Qwen3.5 系として vision input を含むデモを組める
- 単一 GPU で比較的現実的に試しやすい

## ディレクトリ構成

```text
assets/                  # テスト画像
notebooks/               # Jupyter Notebook
outputs/                 # 実行結果(JSON)
scripts/
  generate_test_image.py # 動作確認用画像を生成
  run_demo.sh            # GPU + Docker で一括実行
  text_request.py        # テキスト入力サンプル
  vision_request.py      # 画像入力サンプル
  export_notebook.py     # Notebook生成
```

## セットアップ

### 1. 補助スクリプト用の Python 環境

このリポジトリでは補助スクリプト実行に `uv` を使います。

```bash
cd qwen35-sglang-demo
uv venv .venv
uv pip install -p .venv/bin/python pillow requests nbformat nbconvert jupyter
```

### 2. Hugging Face トークン

必要なら環境変数を設定します。

```bash
export HF_TOKEN=your_huggingface_token
```

### 3. テスト画像生成

```bash
.venv/bin/python scripts/generate_test_image.py
```

## 実行方法

### 一括デモ実行

```bash
bash scripts/run_demo.sh
```

必要ならモデル名やポートを変えられます。

```bash
MODEL_NAME=Qwen/Qwen3.5-2B PORT=30001 bash scripts/run_demo.sh
```

このスクリプトは次を行います。
1. テスト画像を生成
2. `lmsysorg/sglang:latest-cu130-runtime` を使って SGLang サーバを GPU 上で起動
3. `/v1/models` の応答を待機
4. テキスト入力を送信
5. 画像入力を送信
6. 結果を `outputs/` に保存

## Jupyter Notebook

Notebook は以下です。

```text
notebooks/qwen35_sglang_demo.ipynb
```

生成し直すには:

```bash
.venv/bin/python scripts/export_notebook.py
```

Notebook でも次を順に試せます。
- GPU確認
- 画像生成
- DockerでSGLangサーバ起動
- テキスト入力テスト
- 画像入力テスト
- 後始末

## Dockerfile について

このリポジトリの `Dockerfile` は、**補助スクリプトとNotebook実行環境**をまとめたものです。

```bash
docker build -t qwen35-sglang-demo-helper .
```

起動例:

```bash
docker run --rm -it -v $PWD:/app qwen35-sglang-demo-helper bash
```

> SGLang サーバ本体は、Dockerfile 内に再梱包せず **公式 `lmsysorg/sglang:latest-cu130-runtime`** をそのまま利用します。SGLang 本体まで独自 image に焼くより、この方が CUDA / FlashInfer / sglang-kernel の整合性を保ちやすいです。

## 手動起動コマンド

必要ならサーバだけ手動起動できます。

```bash
docker run -d --rm \
  --name qwen35-sglang-demo \
  --gpus all \
  --ipc=host \
  --shm-size 16g \
  -p 30000:30000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $PWD/assets:/workspace/assets \
  -e HF_TOKEN="$HF_TOKEN" \
  lmsysorg/sglang:latest-cu130-runtime \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --context-length 32768 \
    --attention-backend triton \
    --reasoning-parser qwen3
```

リクエスト例:

```bash
.venv/bin/python scripts/text_request.py --model Qwen/Qwen3.5-4B
.venv/bin/python scripts/vision_request.py --model Qwen/Qwen3.5-4B --image assets/demo_image.png
```

## 既知の注意点

- 初回は Docker image pull とモデル download に時間がかかる
- Hugging Face 側の rate limit や gated 設定次第で追加認証が必要
- 利用 GPU メモリが不足する場合は `Qwen/Qwen3.5-2B` に下げる
- 長いコンテキスト長はメモリ消費が大きいので、デモでは `32768` に下げている
- arm64 + CUDA 13 環境では、ホスト直 install より Docker の方が安定しやすい

## 検証チェックリスト

- [x] `nvidia-smi` が見える
- [x] `docker run --gpus all ... nvidia-smi` が通る
- [x] SGLang コンテナが起動する
- [x] `/v1/models` が返る
- [x] テキスト入力が成功する
- [x] 画像入力が成功する
- [ ] Notebook 版でも再現できる
- [x] Dockerfile の helper image が build できる

## 今回の実機検証メモ

- 実行環境 GPU: `NVIDIA GB10`
- 重要: Blackwell 系 GPU では、Qwen3.5 の hybrid GDN モデルに対して `--attention-backend triton` が必要だった
- `flashinfer` のままだと起動時に以下の assertion で失敗した:

```text
AssertionError: triton or trtllm_mha backend are the only supported backends on Blackwell GPUs for hybrid GDN models
```

- 修正後、`Qwen/Qwen3.5-4B` で以下を確認:
  - OpenAI 互換 `/v1/models`
  - テキスト入力推論
  - 画像入力推論

## 出典メモ

- SGLang install docs: CUDA 13 環境では Docker 推奨
- SGLang Qwen3.5 docs: Qwen3.5 対応
- Hugging Face `Qwen/Qwen3.5-4B`: vision encoder を持つことを確認
