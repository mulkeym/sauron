#!/usr/bin/env python3
"""Fail when an OCI image manifest contains an oversized compressed layer."""
from __future__ import annotations

import argparse
import json
import hashlib
import re
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


def check_layout(root: Path, max_bytes: int, platform: str = "linux/amd64") -> dict:
    """Check the actual exported blobs before any registry tags are published."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if json.loads((root / "oci-layout").read_text()).get("imageLayoutVersion") != "1.0.0":
        raise ValueError("unsupported OCI layout")
    layers, platforms, visited = {}, set(), set()

    def blob(descriptor, *, read_json=False):
        digest, size = descriptor.get("digest", ""), descriptor.get("size")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("invalid OCI blob digest")
        if type(size) is not int or size < 0:
            raise ValueError("invalid OCI blob size")
        path = root / "blobs" / "sha256" / digest.split(":")[1]
        if path.is_symlink() or path.stat().st_size != size:
            raise ValueError(f"blob size mismatch: {digest}")
        # Manifests/configs are small. Layer sizes come from actual local files;
        # skopeo verifies their hashes when copying the immutable artifact.
        if read_json:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest.split(":")[1]:
                raise ValueError(f"blob digest mismatch: {digest}")
            return json.loads(raw)
        return size

    def walk(descriptor, depth=0):
        if depth > 8:
            raise ValueError("OCI index nesting exceeds limit")
        manifest = blob(descriptor, read_json=True)
        digest = descriptor["digest"]
        if digest in visited:
            return
        visited.add(digest)
        if "manifests" in manifest:
            for child in manifest["manifests"]:
                walk(child, depth + 1)
            return
        config = blob(manifest["config"], read_json=True)
        platforms.add(f"{config.get('os')}/{config.get('architecture')}")
        check_manifest(manifest, max_bytes)
        for layer in manifest["layers"]:
            layers[layer["digest"]] = blob(layer)

    index = json.loads((root / "index.json").read_text())
    for descriptor in index.get("manifests", []):
        walk(descriptor)
    if platform not in platforms:
        raise ValueError(f"required platform missing: {platform}")
    oversized = [{"digest": d, "size": n} for d, n in layers.items() if n >= max_bytes]
    return {"platforms": sorted(platforms), "layer_count": len(layers),
            "largest_layer_bytes": max(layers.values(), default=0),
            "total_layer_bytes": sum(layers.values()), "oversized": oversized,
            "max_bytes_exclusive": max_bytes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="Manifest JSON or OCI layout directory")
    parser.add_argument("--max-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.max_bytes <= 0:
        parser.error("--max-bytes must be positive")
    if args.manifest.is_dir():
        report = check_layout(args.manifest, args.max_bytes, args.platform)
        oversized = report["oversized"]
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    else:
        oversized = check_manifest(json.loads(args.manifest.read_text()), args.max_bytes)
    if oversized:
        print(f"ERROR: OCI layers must be smaller than {args.max_bytes} bytes")
        return 1
    print(f"All OCI layers are smaller than {args.max_bytes} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
