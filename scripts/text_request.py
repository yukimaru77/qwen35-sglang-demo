import argparse
import json

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-base', default='http://127.0.0.1:30000/v1')
    parser.add_argument('--model', default='Qwen/Qwen3.5-4B')
    parser.add_argument('--prompt', default='Explain in 3 bullet points what SGLang is and why it is useful.')
    parser.add_argument('--max-tokens', type=int, default=256)
    args = parser.parse_args()

    payload = {
        'model': args.model,
        'messages': [{'role': 'user', 'content': args.prompt}],
        'max_tokens': args.max_tokens,
    }
    r = requests.post(f"{args.api_base.rstrip('/')}/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
