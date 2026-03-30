import json
from pathlib import Path

import requests
import sys

sys.path.append(str(ROOT := Path(__file__).resolve().parents[1]))
from scripts.vision_request import image_to_data_url

ASSETS = ROOT / 'assets'
OUT = ROOT / 'outputs' / 'sample_suite_results.json'
API_BASE = 'http://127.0.0.1:30000/v1'
MODEL = 'Qwen/Qwen3.5-4B'

TEXT_PROMPTS = [
    'Explain in 3 bullet points what SGLang is and why it is useful.',
    'Summarize how to serve Qwen3.5 with an OpenAI-compatible API in 4 short bullet points.',
    'Give a concise explanation of why GPU inference frameworks matter for multimodal models.',
]

VISION_CASES = [
    ('sample_shapes.png', 'Describe the image in detail. Count the shapes and mention their colors.'),
    ('sample_chart.png', 'Read this chart. Which quarter is highest, and list the bars in descending order.'),
    ('sample_receipt.png', 'Extract the key information from this receipt, including store, date, and total.'),
]


def chat(messages, max_tokens=256):
    payload = {'model': MODEL, 'messages': messages, 'max_tokens': max_tokens}
    r = requests.post(f'{API_BASE}/chat/completions', json=payload, timeout=900)
    r.raise_for_status()
    return r.json()


def main():
    results = {'text': [], 'vision': []}
    for prompt in TEXT_PROMPTS:
        data = chat([{'role': 'user', 'content': prompt}], max_tokens=256)
        results['text'].append({'prompt': prompt, 'response': data})
    for image_name, prompt in VISION_CASES:
        data = chat([
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': image_to_data_url(ASSETS / image_name)}},
                ],
            }
        ], max_tokens=256)
        results['vision'].append({'image': image_name, 'prompt': prompt, 'response': data})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
