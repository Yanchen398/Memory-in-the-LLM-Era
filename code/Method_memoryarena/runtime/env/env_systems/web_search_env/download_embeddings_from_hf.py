#!/usr/bin/env python3
"""
Download embedding index files from a Hugging Face dataset repo into the local
embeddings directory (e.g. for use with run_search.py and the web_search_env searcher).

Requires: pip install huggingface_hub
Auth: optional for public repos; for private/gated set HF_TOKEN or run `huggingface-cli login`.

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download embedding index files from a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        default="joanna690/websearch-embeddings",
        help="Hugging Face dataset repo (default: joanna690/websearch-embeddings)",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Directory to download into (default: script_dir/embeddings)",
    )
    parser.add_argument(
        "--repo-type",
        choices=("dataset", "model"),
        default="dataset",
        help="Repo type on Hugging Face (default: dataset)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    local_dir = args.local_dir or (script_dir / "embeddings")
    local_dir = local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "Install huggingface_hub: pip install huggingface_hub\n"
            "For private repos, set HF_TOKEN or run: huggingface-cli login"
        )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Downloading {args.repo_id} into {local_dir} ...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    print("Done. Embeddings are in:", local_dir)


if __name__ == "__main__":
    main()
