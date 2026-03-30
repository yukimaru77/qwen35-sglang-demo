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
            md_cell('# Qwen3.5 + SGLang GPU Demo\n\nこの Notebook は、SGLang で Qwen3.5 を GPU 上で起動し、テキスト入力と画像入力の両方を試すデモです。'),
            md_cell('## 1. 前提\n- Docker が使える\n- NVIDIA GPU が見えている\n- Hugging Face のトークンが必要な場合は `HF_TOKEN` を設定する'),
            code_cell('!nvidia-smi'),
            code_cell('!python scripts/generate_test_image.py'),
            md_cell('## 2. SGLang サーバ起動\n下のセルは Docker コンテナをバックグラウンド起動します。必要に応じてモデル名を変更してください。'),
            code_cell("""import os, shlex, subprocess\nMODEL = os.environ.get('MODEL_NAME', 'Qwen/Qwen3.5-4B')\nHF_TOKEN = os.environ.get('HF_TOKEN', '')\ncmd = f'''docker run -d --rm --name qwen35-sglang-demo --gpus all --ipc=host --shm-size 16g -p 30000:30000 -v {Path.cwd() / 'assets'}:/workspace/assets -v {Path.home() / '.cache' / 'huggingface'}:/root/.cache/huggingface -e HF_TOKEN={shlex.quote(HF_TOKEN)} lmsysorg/sglang:latest-cu130-runtime python3 -m sglang.launch_server --model-path {shlex.quote(MODEL)} --host 0.0.0.0 --port 30000 --tp-size 1 --mem-fraction-static 0.8 --context-length 32768 --reasoning-parser qwen3'''\nprint(cmd)\nsubprocess.run(cmd, shell=True, check=False)"""),
            code_cell('!docker logs qwen35-sglang-demo --tail 200'),
            md_cell('## 3. テキスト入力テスト'),
            code_cell('!python scripts/text_request.py --model Qwen/Qwen3.5-4B'),
            md_cell('## 4. 画像入力テスト'),
            code_cell('!python scripts/vision_request.py --model Qwen/Qwen3.5-4B --image assets/demo_image.png'),
            md_cell('## 5. 後始末'),
            code_cell('!docker rm -f qwen35-sglang-demo || true'),
        ],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    out = Path(__file__).resolve().parents[1] / 'notebooks' / 'qwen35_sglang_demo.ipynb'
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
