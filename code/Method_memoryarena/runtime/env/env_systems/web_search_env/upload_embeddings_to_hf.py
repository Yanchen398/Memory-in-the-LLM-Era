#!/usr/bin/env python3
"""
Upload embedding index files from env/env_systems/web_search_env/embeddings/ to Hugging Face.

Uploads:
  - query.pkl
  - shard*.index (FAISS index shards)
  - shard*_id_map.json (docid mapping per shard)

Requires: pip install huggingface_hub
Auth: set HF_TOKEN or run `huggingface-cli login`.

Usage:
  # From repo root or MemoryArena:
  python env/env_systems/web_search_env/upload_embeddings_to_hf.py --repo-id YOUR_USERNAME/browsecomp-plus-embeddings

  # From env_systems/web_search_env:
  python upload_embeddings_to_hf.py --repo-id YOUR_USERNAME/browsecomp-plus-embeddings

  # Create repo on first upload and authenticate with a token:
  HF_TOKEN=your_token python env/env_systems/web_search_env/upload_embeddings_to_hf.py \\
    --repo-id USER/repo --create-repo
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload embedding index files to a Hugging Face repo."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo: USERNAME/REPO_NAME (e.g. myuser/browsecomp-plus-embeddings)",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Local embeddings directory (default: script_dir/embeddings)",
    )
    parser.add_argument(
        "--repo-type",
        choices=("dataset", "model"),
        default="dataset",
        help="Repo type on Hugging Face (default: dataset)",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload FAISS index shards and id maps",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repo if it does not exist (required on first upload)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    embeddings_dir = args.embeddings_dir or (script_dir / "embeddings")
    if not embeddings_dir.is_dir():
        raise SystemExit(f"Embeddings directory not found: {embeddings_dir}")

    try:
        from huggingface_hub import HfApi, create_repo, get_token
    except ImportError:
        raise SystemExit(
            "Install huggingface_hub: pip install huggingface_hub\n"
            "Then set HF_TOKEN or run: huggingface-cli login"
        )

    # Prefer env, then token from huggingface-cli login
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Either:\n"
            "  1. Set HF_TOKEN:  export HF_TOKEN=hf_xxxx\n"
            "  2. Or run:       huggingface-cli login\n"
            "Then run this script again (use --create-repo if the repo does not exist yet)."
        )

    api = HfApi(token=token)

    if args.create_repo:
        try:
            create_repo(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                exist_ok=True,
                token=token,
            )
            print(f"Repo {args.repo_id} ({args.repo_type}) ready.")
        except Exception as e:
            raise SystemExit(f"Failed to create repo: {e}")

    # upload_folder uses LFS for large files automatically
    print(f"Uploading contents of {embeddings_dir} to {args.repo_id} ...")
    try:
        api.upload_folder(
            folder_path=str(embeddings_dir),
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
            token=token,
        )
    except Exception as e:
        err = str(e).lower()
        if "401" in err or "unauthorized" in err or "invalid" in err and "password" in err:
            print(
                "Authentication failed (401). Check your token:\n"
                "  export HF_TOKEN=hf_xxxx   # from https://huggingface.co/settings/tokens\n"
                "  # or: huggingface-cli login",
                file=sys.stderr,
            )
        raise

    print("Done. Files are in:", f"https://huggingface.co/datasets/{args.repo_id}" if args.repo_type == "dataset" else f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
