source .devcontainer/bash_aliases.sh
source .venv/bin/activate

pre-commit install
pre-commit run --all-files

git config alias.s status
git config alias.ss "status -s"

pip install --upgrade pip
