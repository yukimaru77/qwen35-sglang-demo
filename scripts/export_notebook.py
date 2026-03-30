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
            md_cell('# Qwen3.5 + SGLang GPU Demo\n\nこの Notebook は、SGLang で Qwen3.5 を GPU 上で起動し、テキスト入力と画像入力の両方を試すデモです。複数のサンプル画像・サンプルプロンプトも含みます。'),
            md_cell('## 1. 前提\n- Docker が使える\n- NVIDIA GPU が見えている\n- Hugging Face のトークンが必要な場合は `HF_TOKEN` を設定する\n- Blackwell 系 GPU では `--attention-backend triton` が必要'),
            code_cell('!nvidia-smi'),
            code_cell('!python scripts/generate_sample_assets.py'),
            md_cell('## 2. SGLang サーバ起動\n下のセルは Docker コンテナをバックグラウンド起動します。必要に応じてモデル名を変更してください。'),
            code_cell("""from pathlib import Path\nimport os, shlex, subprocess, time, requests\n\nMODEL = os.environ.get('MODEL_NAME', 'Qwen/Qwen3.5-4B')\nHF_TOKEN = os.environ.get('HF_TOKEN', '')\nPROJECT = Path.cwd()\nsubprocess.run('docker rm -f qwen35-sglang-demo >/dev/null 2>&1 || true', shell=True, check=False)\ncmd = f'''docker run -d --rm --name qwen35-sglang-demo --gpus all --ipc=host --shm-size 16g -p 30000:30000 -v {PROJECT / 'assets'}:/workspace/assets -v {Path.home() / '.cache' / 'huggingface'}:/root/.cache/huggingface -e HF_TOKEN={shlex.quote(HF_TOKEN)} lmsysorg/sglang:latest-cu130-runtime python3 -m sglang.launch_server --model-path {shlex.quote(MODEL)} --host 0.0.0.0 --port 30000 --tp-size 1 --mem-fraction-static 0.8 --context-length 32768 --attention-backend triton --reasoning-parser qwen3'''\nprint(cmd)\nsubprocess.run(cmd, shell=True, check=True)\nfor _ in range(180):\n    try:\n        r = requests.get('http://127.0.0.1:30000/v1/models', timeout=5)\n        if r.ok:\n            print('Server is ready')\n            break\n    except Exception:\n        pass\n    time.sleep(5)\nelse:\n    raise RuntimeError('Server did not become ready in time')"""),
            code_cell('!docker logs qwen35-sglang-demo --tail 120'),
            md_cell('## 3. 単発のテキスト入力テスト'),
            code_cell('!python scripts/text_request.py --model Qwen/Qwen3.5-4B'),
            md_cell('## 4. 単発の画像入力テスト'),
            code_cell('!python scripts/vision_request.py --model Qwen/Qwen3.5-4B --image assets/sample_shapes.png'),
            md_cell('## 5. サンプルスイート実行\n複数のテキストプロンプトと画像サンプルを一括実行して JSON 保存します。'),
            code_cell("""import base64, json, mimetypes, requests\nfrom pathlib import Path\n\nROOT = Path.cwd()\nif (ROOT / 'notebooks').exists():\n    PROJECT = ROOT\nelif ROOT.name == 'notebooks':\n    PROJECT = ROOT.parent\nelse:\n    PROJECT = ROOT\n\nMODEL = 'Qwen/Qwen3.5-4B'\nAPI_BASE = 'http://127.0.0.1:30000/v1'\nOUT = PROJECT / 'outputs' / 'sample_suite_results.json'\nOUT.parent.mkdir(parents=True, exist_ok=True)\n\ndef image_to_data_url(path: Path) -> str:\n    mime = mimetypes.guess_type(path.name)[0] or 'image/png'\n    encoded = base64.b64encode(path.read_bytes()).decode('utf-8')\n    return f'data:{mime};base64,{encoded}'\n\ndef chat(messages, max_tokens=256):\n    payload = {'model': MODEL, 'messages': messages, 'max_tokens': max_tokens}\n    r = requests.post(f'{API_BASE}/chat/completions', json=payload, timeout=900)\n    r.raise_for_status()\n    return r.json()\n\ntext_prompts = [\n    'Explain in 3 bullet points what SGLang is and why it is useful.',\n    'Summarize how to serve Qwen3.5 with an OpenAI-compatible API in 4 short bullet points.',\n    'Give a concise explanation of why GPU inference frameworks matter for multimodal models.',\n]\nvision_cases = [\n    (PROJECT / 'assets' / 'sample_shapes.png', 'Describe the image in detail. Count the shapes and mention their colors.'),\n    (PROJECT / 'assets' / 'sample_chart.png', 'Read this chart. Which quarter is highest, and list the bars in descending order.'),\n    (PROJECT / 'assets' / 'sample_receipt.png', 'Extract the key information from this receipt, including store, date, and total.'),\n]\nresults = {'text': [], 'vision': []}\nfor prompt in text_prompts:\n    results['text'].append({'prompt': prompt, 'response': chat([{'role': 'user', 'content': prompt}])})\nfor image_path, prompt in vision_cases:\n    p = Path(image_path)\n    results['vision'].append({\n        'image': p.name,\n        'prompt': prompt,\n        'response': chat([{'role': 'user', 'content': [\n            {'type': 'text', 'text': prompt},\n            {'type': 'image_url', 'image_url': {'url': image_to_data_url(p)}},\n        ]}])\n    })\nOUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')\nprint(OUT)"""),
            code_cell("""import json\nfrom pathlib import Path\nROOT = Path.cwd()\nif (ROOT / 'notebooks').exists():\n    PROJECT = ROOT\nelif ROOT.name == 'notebooks':\n    PROJECT = ROOT.parent\nelse:\n    PROJECT = ROOT\nout = PROJECT / 'outputs' / 'sample_suite_results.json'\nresult = json.loads(out.read_text(encoding='utf-8'))\nprint('text cases:', len(result['text']))\nprint('vision cases:', len(result['vision']))\nprint('first vision image:', result['vision'][0]['image'])\nprint(json.dumps(result['vision'][0]['response']['choices'][0]['message'], ensure_ascii=False, indent=2)[:1200])"""),
            md_cell('## 6. Notebook を保存\n実行済み Notebook を HTML へも変換できます。'),
            code_cell('!jupyter nbconvert --to html notebooks/qwen35_sglang_demo.ipynb --output qwen35_sglang_demo_rendered.html'),
            md_cell('## 7. 後始末'),
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
