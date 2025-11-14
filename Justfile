# https://github.com/casey/just

# Default recipe, it's run when just is invoked without a recipe
default:
  just --list --unsorted

# Sync dev dependencies
dev-sync:
    uv sync --all-extras --cache-dir .uv_cache

# Sync production dependencies (excludes dev dependencies)
prod-sync:
	uv sync --all-extras --no-dev --cache-dir .uv_cache

# Install pre commit hooks
install-hooks:
	uv run pre-commit install

# Run ruff formatting
format:
	uv run ruff format

# Run ruff linting and mypy type checking
lint:
	uv run ruff check --fix
	uv run mypy --ignore-missing-imports --install-types --non-interactive

# Run tests using pytest
test:
	uv run pytest --verbose --color=yes tests

# Run all checks: format, lint, and test
validate: format lint test

# Build docker image
dockerize:
	docker build -t python-repo-template .

download-models:
    @echo "Downloading OpenVoice V2 checkpoints..."
    mkdir -p checkpoints_v2
    wget -q --show-progress https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
    @echo "Extracting checkpoints..."
    unzip -q checkpoints_v2_0417.zip
    rm checkpoints_v2_0417.zip
    @echo "✅ Model checkpoints downloaded and extracted to checkpoints_v2/"

mix:
    repomix
    cat output.md|xclip -i -selection clipboard

tree:
    tree --gitignore

shared:
    uv run shared-cli

mecab:
    uv run python -c "
    import MeCab
    import unidic
    dic_dir = unidic.__path__[0] + '/dicdir'
    tagger = MeCab.Tagger(f'-r /etc/mecab/mecabrc -d {dic_dir}')
    print(tagger.parse('test'))  # Should print parsed output without errors
    "

duber:
    uv run duber-cli test_video2.mp4

download-piper-pt:
    mkdir -p ~/.local/share/piper-voices/pt_BR/cadu/medium
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/cadu/medium/pt_BR-cadu-medium.onnx?download=true \
      -O ~/.local/share/piper-voices/pt_BR/cadu/medium/pt_BR-cadu-medium.onnx
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/cadu/medium/pt_BR-cadu-medium.onnx.json?download=true \
      -O ~/.local/share/piper-voices/pt_BR/cadu/medium/pt_BR-cadu-medium.onnx.json


unidic:
    uv run python -c "
    import unidic
    unidic.download('unidic_lite')
    "

unidic-download:
    uv run python -m unidic download
