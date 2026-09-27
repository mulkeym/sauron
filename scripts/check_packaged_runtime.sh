#!/usr/bin/env bash
# Run against the exact exported image; never mount application source or live data.
set -euo pipefail
image="${1:?usage: check_packaged_runtime.sh IMAGE}"
container="sauron-release-check-${RANDOM}"
cleanup() {
  docker logs "$container" > "${RUNNER_TEMP:-/tmp}/sauron-release-startup.log" 2>&1 || true
  docker rm -fv "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run --rm --network none --entrypoint python "$image" -c '
import importlib, importlib.util, pathlib, shutil, sys
for module in ("src.main", "src.ingestion.emf", "src.ingestion.visio", "src.ingestion.embedding_isolation", "src.mcp.server", "src.sources.service"):
    importlib.import_module(module)
for command in ("inkscape", "vsd2xhtml", "rsvg-convert", "pdftoppm", "tesseract"):
    assert shutil.which(command), command
assert pathlib.Path("/app/.pdf_models_ready").is_file(), "offline models missing"
# Unused base-image tooling must stay out of the runtime (scanner findings).
assert shutil.which("curl") is None, "curl CLI should not be in the runtime image"
base_lib = pathlib.Path(sys.base_prefix) / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}"
system_site = base_lib / "site-packages"
leftover = sorted(p.name for p in (system_site.iterdir() if system_site.is_dir() else []) if p.name.startswith(("pip", "setuptools", "wheel", "pkg_resources")))
assert not leftover, f"system Python tooling left in image: {leftover}"
assert importlib.util.find_spec("pip") is None, "pip should not be in the runtime venv"
assert not list((base_lib / "ensurepip").glob("_bundled/*.whl")), "stale ensurepip wheels"
# Non-root runtime: writable data volume, read-only models and code.
import os
assert os.getuid() != 0 and os.getgid() != 0, f"runs as root: {os.getuid()}:{os.getgid()}"
assert os.access("/app/data", os.W_OK), "/app/data not writable by the app user"
for path in ("/opt/models/huggingface/hub", "/app/src", "/opt/venv"):
    assert os.path.isdir(path) and not os.access(path, os.W_OK), f"{path} should be read-only"
# No outdated native libraries bundled inside Python wheels (scanners that only
# read package metadata cannot see these): OpenSSL 1.x, FFmpeg, Qt 5, and
# libjpeg-turbo 1.x (libjpeg.so.62.0-62.2).
import re
site = pathlib.Path(importlib.util.find_spec("cv2").origin).parents[1]
bundled = [f.name for d in site.glob("*.libs") for f in d.iterdir()]
outdated = [n for n in bundled if re.search(r"^lib(ssl|crypto)\b.*\.so\.1\.|^libav(codec|format|filter|util|device)\b|^libQt5|^libjpeg\b.*\.so\.62\.[0-2]\.", n)]
assert not outdated, f"outdated bundled native libraries: {outdated}"
'
# The offline embedding model (remote code via transformers' module cache)
# must load as the unprivileged user.
docker run --rm --network none --entrypoint python "$image" -c '
import os
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(os.environ["EMBEDDING_MODEL_NAME"], trust_remote_code=True, device="cpu")
assert model.encode(["non-root smoke"]).shape[-1] > 0
'
docker run -d --name "$container" --network none \
  -e API_KEYS=release-smoke-key -e JWT_SECRET_KEY=release-smoke-jwt \
  "$image" >/dev/null
for attempt in $(seq 1 60); do
  if docker exec "$container" python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8080/admin/login", timeout=4)' >/dev/null 2>&1; then
    test "$(docker exec "$container" id -u)" != 0
    docker exec "$container" python -c '
import urllib.request, urllib.error
url="http://127.0.0.1:8080/api/health"
try:
    urllib.request.urlopen(url)
except urllib.error.HTTPError as error:
    assert error.code == 403, error.code
else:
    raise AssertionError("anonymous health request was authorized")
request=urllib.request.Request(url, headers={"X-API-Key":"release-smoke-key"})
assert urllib.request.urlopen(request).status == 200
'
    exit 0
  fi
  sleep 2
done
docker logs "$container"
exit 1
