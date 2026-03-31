# Qwen3.5-27B SGLang OpenAI Quickstart

このリポジトリは、**Docker で Qwen3.5-27B の SGLang API サーバを立てて、Python の OpenAI ライブラリから最速で使う**ための最小構成です。

## 何ができるか

- Docker で `Qwen/Qwen3.5-27B` の API サーバを起動
- OpenAI 互換 API として利用
- Python の `openai` ライブラリから接続
- Notebook からすぐ試せる
- 画像入力（vision input）にも対応

## 前提

- Docker が使えること
- NVIDIA GPU が使えること
- 必要なら Hugging Face トークン `HF_TOKEN` を使えること

## 1. API サーバを起動

### SGLang 版

```bash
docker run -d --rm \
  --name qwen35-sglang-api \
  --gpus all \
  --ipc=host \
  --shm-size 16g \
  -p 30000:30000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  lmsysorg/sglang:latest-cu130-runtime \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-27B \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --context-length 32768 \
    --attention-backend triton \
    --reasoning-parser qwen3
```

### vLLM 版

Blackwell GPU では recipe に従って `cu130-nightly` を使います。

Qwen3 reasoning parser を付けたままでも、**リクエストごとに non-thinking を強制**できます。OpenAI クライアントでは次のように `extra_body.chat_template_kwargs.enable_thinking=False` を渡します。

```python
resp = client.chat.completions.create(
    model='Qwen/Qwen3.5-27B',
    messages=[{'role': 'user', 'content': 'こんにちは。1文で自己紹介して'}],
    extra_body={
        'chat_template_kwargs': {
            'enable_thinking': False
        }
    },
)
```

```bash
docker run -d --rm \
  --name qwen35-vllm-27b \
  --gpus all \
  --ipc=host \
  -p 30010:8000 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:cu130-nightly \
  Qwen/Qwen3.5-27B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching
```

## 2. サーバ確認

### SGLang

```bash
curl http://127.0.0.1:30000/v1/models
```

### vLLM

```bash
curl http://127.0.0.1:30010/v1/models
```

## 3. Python 環境を `uv` で作成

```bash
uv venv .venv
uv sync
```

または依存追加済み環境なら:

```bash
uv sync
```

## 4. OpenAI ライブラリから使う

```python
from openai import OpenAI

client = OpenAI(
    api_key='EMPTY',
    base_url='http://127.0.0.1:30000/v1',
)

resp = client.chat.completions.create(
    model='Qwen/Qwen3.5-27B',
    messages=[
        {'role': 'user', 'content': 'SGLangとは何かを日本語で2文で説明してください。'}
    ],
    max_tokens=64,
)

print(resp)
```

## 5. Notebook を使う

Notebook を生成:

```bash
uv run python scripts/export_notebook.py
```

起動:

```bash
uv run jupyter lab
```

Notebook:

```text
notebooks/openai_quickstart.ipynb
```

## 6. 画像入力の例

まずサンプル画像を生成できます。

```bash
uv run python scripts/generate_sample_image.py
```

その後、以下のように `assets/sample_shapes.png` を使って vision input を試せます。

```python
import base64
import mimetypes
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key='EMPTY',
    base_url='http://127.0.0.1:30000/v1',
)

img = Path('assets/sample_shapes.png')
mime = mimetypes.guess_type(img.name)[0] or 'image/png'
image_url = 'data:' + mime + ';base64,' + base64.b64encode(img.read_bytes()).decode('utf-8')

resp = client.chat.completions.create(
    model='Qwen/Qwen3.5-27B',
    messages=[
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'この画像の図形の数と色を説明してください。'},
                {'type': 'image_url', 'image_url': {'url': image_url}},
            ],
        }
    ],
    max_tokens=64,
)

print(resp)
```

## 7. PDF 入力のサンプル

英語の深層学習論文 PDF を使って、**全ページ** を対象に
- PDF から text を抽出して入力する方法
- PDF を画像化して入力する方法
を Notebook で試せます。

サンプル論文:
- `papers/attention_is_all_you_need.pdf`
- `papers/deep_residual_learning.pdf`

Notebook を生成:

```bash
uv run python scripts/export_pdf_notebook.py
```

Notebook:

```text
notebooks/pdf_input_quickstart.ipynb
```

vLLM 版 PDF notebook を生成:

```bash
uv run python scripts/export_vllm_pdf_notebook.py
```

Notebook:

```text
notebooks/vllm_pdf_input_quickstart.ipynb
```

vLLM 版 PDF non-thinking notebook を生成:

```bash
uv run python scripts/export_vllm_pdf_no_thinking_notebook.py
```

Notebook:

```text
notebooks/vllm_pdf_no_thinking_quickstart.ipynb
```

## 8. GPU メモリ使用量メモ

単一 128GB GPU 環境での、Qwen3.5-27B の概算メモです。

### SGLang

実測では `nvidia-smi` ベースで **約 95.4 GiB** 前後を使用していました。

起動ログの内訳イメージ:
- model weights: 約 53.6 GB
- Mamba cache: 約 16.3 GB
- KV cache (K+V): 約 18.6 GB
- CUDA graph など: 約 1.7 GB
- その他 runtime overhead

### vLLM

vLLM のログ上では概ね以下でした。
- model loading took: **51.1 GiB**
- Available KV cache memory: **40.21 GiB**
- CUDA graph memory: **0.72 GiB actual / 2.83 GiB estimated**

全体感としては、vLLM も **90〜100 GiB クラス** と考えてよいです。

### 補足

- この規模では、GPU メモリの大半は **モデル常駐分** です。
- text / image / PDF-text / PDF-image の違いによる増分は、少なくとも粗い `nvidia-smi` 観測では大きくは見えませんでした。

## 9. 停止

```bash
docker stop qwen35-sglang-api
```

## 補足

- この環境では **`--attention-backend triton`** が必要でした。
- 単一 128GB GPU 環境で `Qwen/Qwen3.5-27B` の起動と API 応答を確認済みです。
- Qwen3.5 は thinking mode が既定のため、返答内容が `reasoning_content` 側に出ることがあります。
