import subprocess
import tempfile


def download_repo(repo: str, base_url: str, dir: str | None = None) -> str:
    """Download a repository from a remote source.

    Args:
        repo (str): The repository name.
        base_url (str): The base URL of the remote repository.
        dir (str, optional): The directory where the repository will be downloaded. If None, a temporary directory will be created.
    If a directory is provided, it will be used instead of creating a temporary one.

    Returns:
        str: The local directory where the repository is downloaded.
    """
    if dir is None:
        dir = tempfile.mkdtemp()
    repo_url = f"{base_url}/{repo}"

    # Clone the repo
    subprocess.run(["git", "clone", repo_url, dir], check=True)

    return dir
