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

## 2. サーバ確認

```bash
curl http://127.0.0.1:30000/v1/models
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

```python
import base64
import mimetypes
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key='EMPTY',
    base_url='http://127.0.0.1:30000/v1',
)

img = Path('image.png')
mime = mimetypes.guess_type(img.name)[0] or 'image/png'
image_url = 'data:' + mime + ';base64,' + base64.b64encode(img.read_bytes()).decode('utf-8')

resp = client.chat.completions.create(
    model='Qwen/Qwen3.5-27B',
    messages=[
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'この画像の内容を説明してください。'},
                {'type': 'image_url', 'image_url': {'url': image_url}},
            ],
        }
    ],
    max_tokens=64,
)

print(resp)
```

## 7. 停止

```bash
docker stop qwen35-sglang-api
```

## 補足

- この環境では **`--attention-backend triton`** が必要でした。
- 単一 128GB GPU 環境で `Qwen/Qwen3.5-27B` の起動と API 応答を確認済みです。
- Qwen3.5 は thinking mode が既定のため、返答内容が `reasoning_content` 側に出ることがあります。
