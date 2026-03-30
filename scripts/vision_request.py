import argparse
import base64
import json
import mimetypes
from pathlib import Path

import requests


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or 'image/png'
    encoded = base64.b64encode(path.read_bytes()).decode('utf-8')
    return f'data:{mime};base64,{encoded}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-base', default='http://127.0.0.1:30000/v1')
    parser.add_argument('--model', default='Qwen/Qwen3.5-4B')
    parser.add_argument('--image', default='assets/demo_image.png')
    parser.add_argument('--prompt', default='Describe this image in detail. Count the shapes and mention their colors.')
    parser.add_argument('--max-tokens', type=int, default=256)
    args = parser.parse_args()

    image_path = Path(args.image)
    payload = {
        'model': args.model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': args.prompt},
                    {'type': 'image_url', 'image_url': {'url': image_to_data_url(image_path)}},
                ],
            }
        ],
        'max_tokens': args.max_tokens,
    }
    r = requests.post(f"{args.api_base.rstrip('/')}/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
