# Qwen3.5 SGLang API Server

このリポジトリは、**Qwen3.5 を SGLang の OpenAI 互換 API サーバとして Docker で起動するための最小メモ**です。

## 前提

- Docker が使えること
- NVIDIA GPU が使えること
- 必要に応じて Hugging Face トークンを使えること

## Qwen3.5-27B を API サーバとして起動

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

## 起動確認

```bash
curl http://127.0.0.1:30000/v1/models
```

## テキスト API テスト

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-27B",
    "messages": [
      {"role": "user", "content": "SGLangとは何かを日本語で2文で説明してください。"}
    ],
    "max_tokens": 64
  }'
```

## 画像入力 API テスト

```bash
IMAGE_BASE64=$(base64 -w 0 image.png)

curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen/Qwen3.5-27B\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"text\", \"text\": \"この画像の内容を説明してください。\"},
          {
            \"type\": \"image_url\",
            \"image_url\": {
              \"url\": \"data:image/png;base64,${IMAGE_BASE64}\"
            }
          }
        ]
      }
    ],
    \"max_tokens\": 64
  }"
```

## ログ確認

```bash
docker logs -f qwen35-sglang-api
```

## 停止

```bash
docker stop qwen35-sglang-api
```

## 補足

- この環境では **`--attention-backend triton`** が必要でした。
- 単一 128GB GPU 環境で、`Qwen/Qwen3.5-27B` の起動と API 応答を確認しました。
- 画像入力も API 経由で動作確認済みです。
