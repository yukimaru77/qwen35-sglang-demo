import json
from pathlib import Path


def md_cell(text: str):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line if line.endswith('\n') else line + '\n' for line in text.splitlines()],
    }


def code_cell(code: str):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line if line.endswith('\n') else line + '\n' for line in code.splitlines()],
    }


def main() -> None:
    nb = {
        'cells': [
            md_cell('# Qwen3.5-27B + SGLang + OpenAI Client Quickstart\n\nこの Notebook は、Docker で SGLang API サーバを起動したあとに、Python の OpenAI ライブラリから Qwen3.5-27B を使う最短手順をまとめたものです。'),
            md_cell('## 1. 前提\n- Docker が使える\n- NVIDIA GPU が使える\n- `HF_TOKEN` が必要に応じて設定されている'),
            code_cell('!docker ps --filter name=qwen35-sglang-api\n!curl -s http://127.0.0.1:30000/v1/models || true'),
            md_cell('## 2. OpenAI クライアント初期化'),
            code_cell("""import os\nfrom openai import OpenAI\n\nBASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:30000/v1')\nclient = OpenAI(\n    api_key='EMPTY',\n    base_url=BASE_URL,\n)\nprint('Using base_url =', BASE_URL)\nclient"""),
            md_cell('## 3. テキスト推論'),
            code_cell("""resp = client.chat.completions.create(\n    model='Qwen/Qwen3.5-27B',\n    messages=[\n        {'role': 'user', 'content': 'SGLangとは何かを日本語で2文で説明してください。'}\n    ],\n    max_tokens=64,\n)\nresp"""),
            md_cell('## 4. 画像入力推論'),
            code_cell("""import base64\nimport mimetypes\nfrom pathlib import Path\n\nimg = Path('image.png')  # ここを自分の画像に置き換える\nif img.exists():\n    mime = mimetypes.guess_type(img.name)[0] or 'image/png'\n    image_url = 'data:' + mime + ';base64,' + base64.b64encode(img.read_bytes()).decode('utf-8')\n\n    resp = client.chat.completions.create(\n        model='Qwen/Qwen3.5-27B',\n        messages=[\n            {\n                'role': 'user',\n                'content': [\n                    {'type': 'text', 'text': 'この画像の内容を説明してください。'},\n                    {'type': 'image_url', 'image_url': {'url': image_url}},\n                ],\n            }\n        ],\n        max_tokens=64,\n    )\n    resp\nelse:\n    print('image.png が見つからないため、このセルはスキップされました。')"""),
            md_cell('## 5. 補足\nQwen3.5 は thinking mode が既定のため、レスポンスは `reasoning_content` 側に現れることがあります。'),
        ],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    out = Path(__file__).resolve().parents[1] / 'notebooks' / 'openai_quickstart.ipynb'
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
