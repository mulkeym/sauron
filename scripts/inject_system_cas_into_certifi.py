#!/usr/bin/env python3
"""Merge the OS CA bundle into certifi's cacert.pem.

Corporate MITM / private roots are installed via update-ca-certificates and
exposed as SSL_CERT_FILE / REQUESTS_CA_BUNDLE. Many Python stacks (requests,
urllib3, huggingface_hub, transformers) still verify with certifi's own file
and ignore the system store. Appending the system bundle fixes those clients.
"""
from __future__ import annotations

import pathlib
import sys

MARKER = b"# --- sauron system CA bundle ---"
SYSTEM_BUNDLE = pathlib.Path("/etc/ssl/certs/ca-certificates.crt")


def main() -> int:
    try:
        import certifi
    except ImportError:
        print("inject_system_cas_into_certifi: certifi not installed; skip", file=sys.stderr)
        return 0

    if not SYSTEM_BUNDLE.is_file() or SYSTEM_BUNDLE.stat().st_size == 0:
        print(f"inject_system_cas_into_certifi: missing {SYSTEM_BUNDLE}", file=sys.stderr)
        return 1

    certifi_path = pathlib.Path(certifi.where())
    if certifi_path.resolve() == SYSTEM_BUNDLE.resolve():
        print(
            "inject_system_cas_into_certifi: certifi already uses the system CA bundle"
        )
        return 0

    existing = certifi_path.read_bytes()
    system = SYSTEM_BUNDLE.read_bytes()
    # Refresh the managed section on every build. This matters when an existing
    # venv is copied into a runtime stage whose OS bundle contains newer or
    # site-specific roots.
    certifi_base = existing.split(MARKER, 1)[0].rstrip()
    merged = certifi_base + b"\n" + MARKER + b"\n" + system + b"\n"
    if existing == merged:
        print(
            "inject_system_cas_into_certifi: current system bundle already "
            f"merged into {certifi_path}"
        )
        return 0

    certifi_path.write_bytes(merged)
    print(
        f"inject_system_cas_into_certifi: appended {SYSTEM_BUNDLE} "
        f"({len(system)} bytes) -> {certifi_path} "
        f"(now {certifi_path.stat().st_size} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
