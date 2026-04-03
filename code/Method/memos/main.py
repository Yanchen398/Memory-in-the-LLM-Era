import asyncio

from .configuration import build_runtime_config, load_runtime_config_file
from .locomo_ingestion import ingestion
from .locomo_responses import response
from .locomo_search import search


def run_memos(
    version="default",
    num_workers=4,
    top_k=20,
    dataset_path="/home/docker/IndepthMem/Dataset/LOCOMO/locomodemo.json",
    config=None,
    config_path=None,
):
    if config is None and config_path:
        config = load_runtime_config_file(config_path)

    runtime_config = build_runtime_config(
        config
        or {
            "version": version,
            "num_workers": num_workers,
            "top_k": top_k,
            "dataset_path": dataset_path,
        },
        config_path=config_path,
    )
    ingestion(runtime_config)
    search(runtime_config)
    asyncio.run(response(runtime_config))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for saving results (e.g., 1010)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers to process users"
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Number of results to retrieve in search queries"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/docker/IndepthMem/Dataset/LOCOMO/locomodemo.json",
        help="Path to the LOCOMO dataset JSON file",
    )
    parser.add_argument("--config_path", type=str, help="Path to the memos config file")
    args = parser.parse_args()

    run_memos(
        version=args.version,
        num_workers=args.workers,
        top_k=args.top_k,
        dataset_path=args.dataset_path,
        config_path=args.config_path,
    )
