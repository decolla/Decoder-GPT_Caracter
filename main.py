try:
    from transformers import pipeline
except ModuleNotFoundError as error:
    raise SystemExit(
        "Failed to import a required dependency for transformers.\n"
        "Please activate your project virtual environment and reinstall dependencies:\n\n"
        "  source .venv/bin/activate\n"
        "  python -m pip install --upgrade pip\n"
        "  python -m pip install --force-reinstall regex transformers sentencepiece torch\n"
    ) from error


def main():
    generator = pipeline("text-generation")

    text = "6 de maio de 2026"
    result = generator(text)

    print(result)


if __name__ == "__main__":
    main()
