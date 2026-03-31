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
            md_cell('# PDF Input Quickstart for Qwen3.5-27B on vLLM\n\nこの Notebook は、英語論文 PDF の **全ページ** を対象にして、vLLM の OpenAI 互換 API 経由で **text として入れる方法** と **画像化して vision input として入れる方法** をまとめたものです。'),
            md_cell('## 1. 前提\n- Docker で vLLM API サーバが起動している\n- `OPENAI_BASE_URL` を必要に応じて設定する\n- PDF サンプルは `papers/` に配置済み'),
            code_cell("""import os\nfrom pathlib import Path\nfrom openai import OpenAI\n\nBASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')\nPAPERS = Path('../papers')\nprint('Using base_url =', BASE_URL)\nprint('Papers:', sorted(p.name for p in PAPERS.glob('*.pdf')))\nclient = OpenAI(api_key='EMPTY', base_url=BASE_URL, timeout=3600)\nclient"""),
            md_cell('## 2. PDF の全ページから text を抽出して入れる'),
            code_cell("""from pypdf import PdfReader\n\npdf_path = PAPERS / 'attention_is_all_you_need.pdf'\nreader = PdfReader(str(pdf_path))\ntext = '\\n'.join(page.extract_text() or '' for page in reader.pages)\nprint('page_count =', len(reader.pages))\nprint('text_chars =', len(text))\nprint(text[:3000])"""),
            code_cell("""resp = client.chat.completions.create(\n    model='Qwen/Qwen3.5-27B',\n    messages=[\n        {'role': 'system', 'content': 'You are reading a full English deep learning paper extracted from PDF text.'},\n        {'role': 'user', 'content': 'Summarize the paper in Japanese with sections for problem, method, and key ideas.\\n\\n' + text[:60000]},\n    ],\n    max_tokens=512,\n)\nresp"""),
            md_cell('## 3. PDF の全ページを画像化して入れる'),
            code_cell("""import fitz  # pymupdf\n\npdf_path = PAPERS / 'deep_residual_learning.pdf'\ndoc = fitz.open(pdf_path)\nout_dir = PAPERS / 'rendered' / 'deep_residual_learning_all_pages_vllm'\nout_dir.mkdir(parents=True, exist_ok=True)\nimage_paths = []\nfor i in range(len(doc)):\n    page = doc.load_page(i)\n    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))\n    img_path = out_dir / f'deep_residual_learning_page_{i+1:02d}.png'\n    pix.save(img_path)\n    image_paths.append(img_path)\nprint('rendered_pages =', len(image_paths))\nprint(image_paths[:3], '...')"""),
            code_cell("""import base64\nimport mimetypes\n\ncontent = [{'type': 'text', 'text': 'These are rendered pages from an English deep learning paper PDF. Summarize the paper in Japanese and mention the overall topic and architecture.'}]\nfor img_path in image_paths:\n    mime = mimetypes.guess_type(img_path.name)[0] or 'image/png'\n    image_url = 'data:' + mime + ';base64,' + base64.b64encode(img_path.read_bytes()).decode('utf-8')\n    content.append({'type': 'image_url', 'image_url': {'url': image_url}})\n\nresp = client.chat.completions.create(\n    model='Qwen/Qwen3.5-27B',\n    messages=[{'role': 'user', 'content': content}],\n    max_tokens=512,\n)\nresp"""),
            md_cell('## 4. 補足\n- text 抽出は本文全体の要約向き\n- 画像化は図表・レイアウト・数式を含めた理解向き'),
        ],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    out = Path(__file__).resolve().parents[1] / 'notebooks' / 'vllm_pdf_input_quickstart.ipynb'
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
