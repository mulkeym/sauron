"""Release gate checks every blob, including attestations, before publication."""
import hashlib
import json

import pytest

from scripts.check_oci_layer_sizes import check_layout


def layout(tmp_path, layer_size=9, architecture="amd64", attestation_size=5):
    (tmp_path / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    blobs = tmp_path / "blobs/sha256"
    blobs.mkdir(parents=True)

    def put(value):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        digest = hashlib.sha256(raw).hexdigest()
        (blobs / digest).write_bytes(raw)
        return {"digest": "sha256:" + digest, "size": len(raw)}

    def manifest(arch, size):
        return put({"schemaVersion": 2,
                    "config": put({"os": "linux", "architecture": arch}),
                    "layers": [put(b"x" * size)]})

    root = put({"schemaVersion": 2, "manifests": [
        manifest(architecture, layer_size), manifest("unknown", attestation_size)]})
    (tmp_path / "index.json").write_text(json.dumps({"manifests": [root]}))
    return root


def test_nested_layout_includes_attestations_and_accepts_under_limit(tmp_path):
    layout(tmp_path)
    report = check_layout(tmp_path, 10)
    assert report["largest_layer_bytes"] == 9
    assert report["layer_count"] == 2
    assert report["total_layer_bytes"] == 14
    assert report["oversized"] == []


@pytest.mark.parametrize("sizes", [(10, 5), (9, 10), (11, 5)])
def test_rejects_boundary_and_oversized_attestations(tmp_path, sizes):
    layout(tmp_path, layer_size=sizes[0], attestation_size=sizes[1])
    assert check_layout(tmp_path, 10)["oversized"]


def test_requires_production_architecture(tmp_path):
    layout(tmp_path, architecture="arm64")
    with pytest.raises(ValueError, match="required platform"):
        check_layout(tmp_path, 10)


@pytest.mark.parametrize("damage", ["missing", "size", "digest"])
def test_fails_closed_on_damaged_manifest(tmp_path, damage):
    descriptor = layout(tmp_path)
    path = tmp_path / "blobs/sha256" / descriptor["digest"].split(":")[1]
    if damage == "missing":
        path.unlink()
    elif damage == "size":
        path.write_bytes(b"broken")
    else:
        path.write_bytes(b" " * path.stat().st_size)
    with pytest.raises((ValueError, FileNotFoundError)):
        check_layout(tmp_path, 10)
