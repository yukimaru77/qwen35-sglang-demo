from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    print('Qwen3.5 + SGLang GPU demo project')
    print(f'Project root: {root}')
    print('Read README.md for setup and execution instructions.')


if __name__ == '__main__':
    main()
