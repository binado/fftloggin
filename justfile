build-docs:
    uv run --group docs sphinx-build -E -a -b html -W --keep-going docs docs/_build/html
