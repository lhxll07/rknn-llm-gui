"""Download a public ModelScope model with per-file SHA256 validation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = quote(args.repo, safe="/")
    revision = quote(args.revision, safe="")
    listing = f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision={revision}&Recursive=true"
    with urlopen(listing, timeout=60) as response:
        metadata = json.load(response)
    if metadata.get("Code") != 200:
        raise ValueError("ModelScope did not return a valid file listing")
    files = [item for item in metadata["Data"]["Files"] if item.get("Type") != "tree"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "modelscope-source.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def download(item):
        name = item["Path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid model file path")
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        url = f"https://www.modelscope.cn/models/{repo}/resolve/{revision}/{quote(name, safe='/')}"
        digest = hashlib.sha256()
        count = 0
        last = time.monotonic()
        with urlopen(url, timeout=90) as response, partial.open("xb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                count += len(block)
                if time.monotonic() - last >= 20:
                    print(f"Downloading {name}: {count}/{item['Size']} bytes", flush=True)
                    last = time.monotonic()
        if count != item["Size"] or digest.hexdigest() != item["Sha256"]:
            raise ValueError(f"Size or SHA256 mismatch: {name}; partial file kept for diagnosis")
        partial.rename(path)
        print(f"Verified {name}: {count} bytes", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        for _ in pool.map(download, files):
            pass


if __name__ == "__main__":
    main()
