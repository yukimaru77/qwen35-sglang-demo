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
            md_cell('# PDF Input Quickstart for Qwen3.5-27B\n\nこの Notebook は、英語論文 PDF を **text として入れる方法** と **画像化して vision input として入れる方法** をまとめたものです。サンプルとして arXiv の深層学習系論文 2 本を使います。'),
            md_cell('## 1. 前提\n- Docker で SGLang API サーバが起動している\n- `OPENAI_BASE_URL` を必要に応じて設定する\n- PDF サンプルは `papers/` に配置済み'),
            code_cell("""import os\nfrom pathlib import Path\nfrom openai import OpenAI\n\nBASE_URL = os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:30000/v1')\nPAPERS = Path('../papers')\nprint('Using base_url =', BASE_URL)\nprint('Papers:', sorted(p.name for p in PAPERS.glob('*.pdf')))\nclient = OpenAI(api_key='EMPTY', base_url=BASE_URL)\nclient"""),
            md_cell('## 2. PDF から text だけ抽出して入れる'),
            code_cell("""from pypdf import PdfReader\n\npdf_path = PAPERS / 'attention_is_all_you_need.pdf'\nreader = PdfReader(str(pdf_path))\ntext = '\\n'.join(page.extract_text() or '' for page in reader.pages[:3])\nprint(text[:3000])"""),
            code_cell("""resp = client.chat.completions.create(\n    model='Qwen/Qwen3.5-27B',\n    messages=[\n        {'role': 'system', 'content': 'You are reading an English deep learning paper extracted from PDF text.'},\n        {'role': 'user', 'content': 'Summarize the following paper excerpt in Japanese in 5 bullet points.\\n\\n' + text[:12000]},\n    ],\n    max_tokens=256,\n)\nresp"""),
            md_cell('## 3. PDF を画像化して vision input として入れる'),
            code_cell("""import fitz  # pymupdf\n\npdf_path = PAPERS / 'deep_residual_learning.pdf'\ndoc = fitz.open(pdf_path)\npage = doc.load_page(0)\npix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))\nout_dir = Path('../papers/rendered')\nout_dir.mkdir(parents=True, exist_ok=True)\nimg_path = out_dir / 'deep_residual_learning_page1.png'\npix.save(img_path)\nimg_path"""),
            code_cell("""import base64\nimport mimetypes\n\nmime = mimetypes.guess_type(img_path.name)[0] or 'image/png'\nimage_url = 'data:' + mime + ';base64,' + base64.b64encode(img_path.read_bytes()).decode('utf-8')\n\nresp = client.chat.completions.create(\n    model='Qwen/Qwen3.5-27B',\n    messages=[\n        {\n            'role': 'user',\n            'content': [\n                {'type': 'text', 'text': 'This is the first page of an English deep learning paper PDF rendered as an image. Summarize what kind of paper it is in Japanese.'},\n                {'type': 'image_url', 'image_url': {'url': image_url}},\n            ],\n        }\n    ],\n    max_tokens=256,\n)\nresp"""),
            md_cell('## 4. 補足\n- text 抽出は本文検索・要約に向く\n- 画像化は図表・レイアウト・数式を含めて見せたい時に向く'),
        ],
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    out = Path(__file__).resolve().parents[1] / 'notebooks' / 'pdf_input_quickstart.ipynb'
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
