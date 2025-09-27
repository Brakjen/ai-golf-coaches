from pathlib import Path


def get_git_root() -> Path:
    return Path(__file__).parent.parent.resolve()
