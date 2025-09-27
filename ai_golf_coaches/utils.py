from pathlib import Path


def get_git_root() -> Path:
    """Get the root directory of the git repository.

    Returns:
        Path: The root directory of the git repository.

    """
    return Path(__file__).parent.parent.resolve()
