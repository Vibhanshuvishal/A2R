from __future__ import annotations

import argparse
import time

from a2r.settings import load_config, project_path
from a2r.storage import VectorStoreManager


def main():
    parser = argparse.ArgumentParser(description="Ingest A2R local knowledge bases")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    store = VectorStoreManager(config)
    for pipeline in config["pipelines"]:
        started = time.monotonic()
        count = store.ingest_directory(pipeline["id"], project_path(pipeline["data_dir"]))
        print(f"{pipeline['name']}: {count} new chunks in {time.monotonic() - started:.2f}s")


if __name__ == "__main__":
    main()
