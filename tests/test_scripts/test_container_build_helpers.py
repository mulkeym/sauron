from __future__ import annotations

import os
import shutil
import sys
import types
from pathlib import Path

import pytest

from scripts.check_oci_layer_sizes import check_manifest
from scripts.configure_hf_tls import configure_huggingface_tls
from scripts.split_layer_tree import split_tree


def _merge_layers(output: Path, destination: Path) -> None:
    for layer in sorted(path for path in output.iterdir() if path.is_dir()):
        shutil.copytree(layer, destination, dirs_exist_ok=True, symlinks=True)


def test_split_tree_reconstructs_files_links_and_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a").mkdir()
    (source / "b").mkdir()
    (source / "a" / "large.bin").write_bytes(b"a" * 70)
    (source / "b" / "other.bin").write_bytes(b"b" * 60)
    os.link(source / "a" / "large.bin", source / "b" / "hardlink.bin")
    (source / "link").symlink_to("a/large.bin")

    output = tmp_path / "layers"
    manifest = split_tree(source, output, layer_count=3, max_bytes=80)
    assert max(layer["bytes"] for layer in manifest["layers"]) <= 80
    assert manifest["total_bytes"] == 130
    hardlink_layer = next(
        layer
        for layer in output.iterdir()
        if layer.is_dir() and (layer / "a" / "large.bin").exists()
    )
    assert os.stat(hardlink_layer / "a" / "large.bin").st_ino == os.stat(
        hardlink_layer / "b" / "hardlink.bin"
    ).st_ino

    restored = tmp_path / "restored"
    restored.mkdir()
    _merge_layers(output, restored)
    assert (restored / "a" / "large.bin").read_bytes() == b"a" * 70
    assert (restored / "b" / "other.bin").read_bytes() == b"b" * 60
    assert (restored / "link").is_symlink()
    assert (restored / "b" / "hardlink.bin").read_bytes() == b"a" * 70


def test_split_tree_rejects_a_single_oversized_file(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "too-large").write_bytes(b"x" * 81)
    with pytest.raises(ValueError, match="single file exceeds"):
        split_tree(source, tmp_path / "layers", layer_count=2, max_bytes=80)


def test_oci_manifest_layer_limit():
    manifest = {
        "layers": [
            {"digest": "sha256:small", "size": 999},
            {"digest": "sha256:limit", "size": 1000},
        ]
    }
    assert check_manifest(manifest, 1000) == [
        {"index": 1, "size": 1000, "digest": "sha256:limit"}
    ]


def test_huggingface_v1_client_uses_explicit_enterprise_ca(tmp_path, monkeypatch):
    ca_bundle = tmp_path / "enterprise-ca.pem"
    ca_bundle.write_text("certificate data")
    captured = {}

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.close_session = lambda: captured.setdefault("closed", True)
    fake_hub.set_client_factory = lambda factory: captured.setdefault("sync", factory)
    fake_hub.set_async_client_factory = lambda factory: captured.setdefault("async", factory)

    class FakeTimeout:
        def __init__(self, value):
            self.value = value

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Timeout = FakeTimeout
    fake_httpx.Client = FakeClient
    fake_httpx.AsyncClient = FakeClient
    ssl_context = object()

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        monkeypatch.setenv(key, "before-test")
    monkeypatch.setattr(
        "scripts.configure_hf_tls.ssl.create_default_context",
        lambda *, cafile: ssl_context if cafile == str(ca_bundle) else None,
    )

    backend = configure_huggingface_tls(str(ca_bundle))
    client = captured["sync"]()
    async_client = captured["async"]()

    assert backend == "huggingface_hub 1.x httpx client factory"
    assert captured["closed"] is True
    assert client.kwargs["verify"] is ssl_context
    assert async_client.kwargs["verify"] is ssl_context
    assert client.kwargs["trust_env"] is True
    assert os.environ["SSL_CERT_FILE"] == str(ca_bundle)
    assert os.environ["REQUESTS_CA_BUNDLE"] == str(ca_bundle)
