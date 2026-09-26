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
'
docker run -d --name "$container" --network none \
  -e API_KEYS=release-smoke-key -e JWT_SECRET_KEY=release-smoke-jwt \
  "$image" >/dev/null
for attempt in $(seq 1 60); do
  if docker exec "$container" python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8080/admin/login", timeout=4)' >/dev/null 2>&1; then
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
