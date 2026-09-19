#!/usr/bin/env python3
"""Fail when an OCI image manifest contains an oversized compressed layer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_manifest(manifest: dict, max_bytes: int) -> list[dict]:
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("input is not an OCI image manifest with layers")
    oversized = []
    for index, layer in enumerate(layers):
        size = int(layer.get("size", -1))
        if size < 0:
            raise ValueError(f"layer {index} has no valid size")
        digest = str(layer.get("digest", "unknown"))
        print(f"OCI layer {index:02d}: {size} bytes ({size / 1_000_000:.1f} MB) {digest}")
        if size >= max_bytes:
            oversized.append({"index": index, "size": size, "digest": digest})
    return oversized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    oversized = check_manifest(manifest, args.max_bytes)
    if oversized:
        details = ", ".join(
            f"layer {item['index']}={item['size']}" for item in oversized
        )
        print(f"ERROR: OCI layers must be smaller than {args.max_bytes} bytes: {details}")
        return 1
    print(f"All OCI layers are smaller than {args.max_bytes} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
