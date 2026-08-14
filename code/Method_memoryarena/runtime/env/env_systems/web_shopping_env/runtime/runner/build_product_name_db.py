#!/usr/bin/env python3
"""Build a compact ASIN-to-name SQLite index without loading the catalog."""

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterator


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        first = stream.read(1)
        if first != "[":
            raise ValueError(f"Expected a JSON array in {path}")

        buffer = ""
        eof = False
        while True:
            if not eof and len(buffer) < chunk_size:
                chunk = stream.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            if not buffer and eof:
                raise ValueError(f"Unexpected end of JSON array in {path}")

            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = stream.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue

            if isinstance(item, dict):
                yield item
            buffer = buffer[end:]


def database_is_current(output_path: Path, source_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        connection = sqlite3.connect(f"file:{output_path}?mode=ro", uri=True)
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        connection.close()
    except (OSError, sqlite3.Error, ValueError):
        return False
    stat = source_path.stat()
    return (
        metadata.get("source_size") == str(stat.st_size)
        and metadata.get("source_mtime_ns") == str(stat.st_mtime_ns)
        and int(metadata.get("product_count", "0")) > 0
    )


def build_database(source_path: Path, output_path: Path) -> None:
    if database_is_current(output_path, source_path):
        print(f"product_name_db_ready path={output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
    connection = sqlite3.connect(tmp_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        "CREATE TABLE products (asin TEXT PRIMARY KEY, name TEXT NOT NULL) WITHOUT ROWID"
    )
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    batch = []
    parsed_count = 0
    named_count = 0
    for product in iter_json_array(source_path):
        parsed_count += 1
        asin = product.get("asin")
        if not asin:
            product_info = product.get("product_information") or {}
            asin = product_info.get("ASIN")
        name = product.get("name") or product.get("title") or product.get("small_description")
        if asin and name:
            batch.append((str(asin).strip(), str(name).strip()))
            named_count += 1
        if len(batch) >= 5000:
            connection.executemany(
                "INSERT OR REPLACE INTO products (asin, name) VALUES (?, ?)", batch
            )
            batch.clear()
        if parsed_count % 100000 == 0:
            connection.commit()
            print(f"product_name_db_progress parsed={parsed_count} named={named_count}", flush=True)

    if batch:
        connection.executemany(
            "INSERT OR REPLACE INTO products (asin, name) VALUES (?, ?)", batch
        )
    stat = source_path.stat()
    metadata = {
        "source_path": str(source_path),
        "source_size": str(stat.st_size),
        "source_mtime_ns": str(stat.st_mtime_ns),
        "parsed_count": str(parsed_count),
        "product_count": str(named_count),
    }
    connection.executemany(
        "INSERT INTO metadata (key, value) VALUES (?, ?)", metadata.items()
    )
    connection.commit()
    connection.close()
    os.replace(tmp_path, output_path)
    print(
        f"product_name_db_ready path={output_path} parsed={parsed_count} named={named_count}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_database(args.source, args.output)


if __name__ == "__main__":
    main()
