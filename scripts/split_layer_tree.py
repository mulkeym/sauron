#!/usr/bin/env python3
"""Partition a filesystem tree into deterministic, size-bounded overlay trees.

Each output directory can be copied into the same container destination by a
separate Dockerfile COPY instruction. Regular files that are hard-linked in the
source remain in the same partition and retain their hard links.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from collections import defaultdict
from pathlib import Path


def _entries(source: Path):
    directories: list[Path] = []
    symlinks: list[Path] = []
    file_groups: dict[tuple[int, int], list[Path]] = defaultdict(list)

    for root, dirnames, filenames in os.walk(source, topdown=True, followlinks=False):
        root_path = Path(root)
        for name in list(dirnames):
            path = root_path / name
            if path.is_symlink():
                symlinks.append(path)
                dirnames.remove(name)
            else:
                directories.append(path)
        for name in filenames:
            path = root_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                symlinks.append(path)
            elif stat.S_ISREG(info.st_mode):
                file_groups[(info.st_dev, info.st_ino)].append(path)
            else:
                raise ValueError(f"unsupported filesystem entry: {path}")

    return directories, symlinks, list(file_groups.values())


def split_tree(source: Path, output: Path, *, layer_count: int, max_bytes: int) -> dict:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError(f"source is not a directory: {source}")
    if source == output or source in output.parents:
        raise ValueError("output must not be inside source")
    if layer_count < 1 or max_bytes < 1:
        raise ValueError("layer_count and max_bytes must be positive")

    if output.exists():
        shutil.rmtree(output)
    layers = [output / f"{index:02d}" for index in range(layer_count)]
    for layer in layers:
        layer.mkdir(parents=True)

    directories, symlinks, groups = _entries(source)
    totals = [0] * layer_count
    counts = [0] * layer_count
    assignments: list[tuple[int, list[Path]]] = []

    sized_groups = []
    for paths in groups:
        size = paths[0].stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"single file exceeds layer limit ({size} > {max_bytes}): "
                f"{paths[0].relative_to(source)}"
            )
        sized_groups.append((size, sorted(paths)))

    for size, paths in sorted(
        sized_groups,
        key=lambda item: (-item[0], str(item[1][0].relative_to(source))),
    ):
        candidates = sorted(range(layer_count), key=lambda index: (totals[index], index))
        layer_index = next(
            (index for index in candidates if totals[index] + size <= max_bytes),
            None,
        )
        if layer_index is None:
            total = sum(totals) + size
            raise ValueError(
                f"tree needs more partitions: {total} bytes cannot fit in "
                f"{layer_count} layers capped at {max_bytes} bytes"
            )
        totals[layer_index] += size
        counts[layer_index] += len(paths)
        assignments.append((layer_index, paths))

    # Empty directories and symlinks contain no material payload. Keeping them
    # in layer zero makes the partitioning stable and preserves the full tree.
    for directory in sorted(directories):
        relative = directory.relative_to(source)
        target = layers[0] / relative
        target.mkdir(parents=True, exist_ok=True)
        shutil.copystat(directory, target, follow_symlinks=False)

    for link in sorted(symlinks):
        relative = link.relative_to(source)
        target = layers[0] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.readlink(link))
        counts[0] += 1

    for layer_index, paths in assignments:
        first_target: Path | None = None
        for path in paths:
            relative = path.relative_to(source)
            target = layers[layer_index] / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if first_target is None:
                try:
                    os.link(path, target, follow_symlinks=False)
                except OSError:
                    shutil.copy2(path, target, follow_symlinks=False)
                first_target = target
            else:
                os.link(first_target, target, follow_symlinks=False)

    manifest = {
        "source": str(source),
        "layer_count": layer_count,
        "max_bytes": max_bytes,
        "total_bytes": sum(totals),
        "layers": [
            {"name": layer.name, "bytes": totals[index], "entries": counts[index]}
            for index, layer in enumerate(layers)
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer-count", type=int, default=16)
    parser.add_argument("--max-bytes", type=int, default=850_000_000)
    args = parser.parse_args()
    manifest = split_tree(
        args.source,
        args.output,
        layer_count=args.layer_count,
        max_bytes=args.max_bytes,
    )
    for layer in manifest["layers"]:
        print(
            f"split_layer_tree: {layer['name']} "
            f"bytes={layer['bytes']} entries={layer['entries']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
