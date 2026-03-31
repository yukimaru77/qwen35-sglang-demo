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
            md_cell('# Qwen3.5-27B + vLLM Non-Thinking Quickstart\n\nこの Notebook は、vLLM の Qwen3 reasoning parser を使ったまま、**リクエストごとに `enable_thinking=False` を渡して non-thinking モード**で使う最小例です。'),
            md_cell('## 1. 前提\n- vLLM API サーバが `--reasoning-parser qwen3` 付きで起動している\n- `OPENAI_BASE_URL` を必要に応じて設定する'),
            code_cell("""import os\nfrom openai import OpenAI\n\nBASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')\nclient = OpenAI(api_key='EMPTY', base_url=BASE_URL, timeout=3600)\nprint('Using base_url =', BASE_URL)\nclient"""),
            md_cell('## 2. テキスト non-thinking 推論'),
            code_cell("""resp = client.chat.completions.create(\n    model='Qwen/Qwen3.5-27B',\n    messages=[\n        {'role': 'user', 'content': 'こんにちは。1文で自己紹介して'}\n    ],\n    extra_body={\n        'chat_template_kwargs': {\n            'enable_thinking': False\n        }\n    },\n    max_tokens=64,\n)\nresp"""),
            md_cell('## 3. 画像入力 non-thinking 推論'),
            code_cell("""import base64\nimport mimetypes\nfrom pathlib import Path\n\nimg = Path('../assets/sample_shapes.png')\nmime = mimetypes.guess_type(img.name)[0] or 'image/png'\nimage_url = 'data:' + mime + ';base64,' + base64.b64encode(img.read_bytes()).decode('utf-8')\n\nresp = client.chat.completions.create(\n    model='Qwen/Qwen3.5-27B',\n    messages=[\n        {\n            'role': 'user',\n            'content': [\n                {'type': 'text', 'text': 'この画像の図形の数と色を1文で説明してください。'},\n                {'type': 'image_url', 'image_url': {'url': image_url}},\n            ]\n        }\n    ],\n    extra_body={\n        'chat_template_kwargs': {\n            'enable_thinking': False\n        }\n    },\n    max_tokens=64,\n)\nresp"""),
        ],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    out = Path(__file__).resolve().parents[1] / 'notebooks' / 'vllm_no_thinking_quickstart.ipynb'
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
