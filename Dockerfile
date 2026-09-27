# Base image: Wolfi (glibc, rolling, continuously patched). Debian's slim
# images carried ~300 Critical/High/Medium OS findings with no fix available;
# the same package set on Wolfi scans clean. Override with a digest-pinned
# mirror copy for reproducible/air-gapped builds. Pinned by digest (bump it
# periodically). The runtime stage runs `apk upgrade`, so every shipped package
# comes from the current Wolfi repository with the latest security fixes.
ARG WOLFI_IMAGE=cgr.dev/chainguard/wolfi-base@sha256:08df5982c3d27e70a4ce1607e3bb9af09d746f8722cf135a7694afef879fc5a2

# Stage 0: native document renderers that Wolfi does not package:
#   - Inkscape 1.4.x (EMF -> PNG), with the gtkmm-3 C++ binding stack it needs
#   - librevenge + libvisio tools (vsd2xhtml: Visio -> SVG)
# Everything installs under /opt/inkscape and /opt/libvisio with an RPATH, so
# the runtime stage only copies these trees plus Wolfi's shared libraries.
# Sources are pinned by SHA-256. Bump versions and hashes together.
FROM ${WOLFI_IMAGE} AS native-tools

# Parallel compile jobs; defaults to the builder's CPU count.
ARG BUILD_JOBS=

RUN apk add --no-cache \
      build-base cmake samurai meson pkgconf perl gperf python-3.11 curl xz bzip2 \
      boost-dev icu-dev libxml2-dev libxslt-dev zlib-dev \
      glib-dev cairo-dev pango-dev gtk-3-dev gdk-pixbuf-dev harfbuzz-dev \
      fontconfig-dev freetype-dev libepoxy-dev \
      gsl-dev gc-dev double-conversion-dev lcms2-dev libpng-dev \
      libjpeg-turbo-dev potrace-dev readline-dev

WORKDIR /src
COPY docker/native-sources.sha256 /src/SHA256SUMS
RUN set -eu; \
    for url in \
      https://download.gnome.org/sources/libsigc++/2.12/libsigc++-2.12.1.tar.xz \
      https://download.gnome.org/sources/glibmm/2.66/glibmm-2.66.10.tar.xz \
      https://www.cairographics.org/releases/cairomm-1.14.6.tar.xz \
      https://download.gnome.org/sources/pangomm/2.46/pangomm-2.46.5.tar.xz \
      https://download.gnome.org/sources/atkmm/2.28/atkmm-2.28.5.tar.xz \
      https://download.gnome.org/sources/gtkmm/3.24/gtkmm-3.24.11.tar.xz \
      https://media.inkscape.org/dl/resources/file/inkscape-1.4.4.tar.xz \
      https://dev-www.libreoffice.org/src/librevenge-0.0.5.tar.bz2 \
      https://dev-www.libreoffice.org/src/libvisio-0.1.8.tar.xz \
      https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.tar.bz2; do \
      curl -fsSL --retry 3 -o "$(basename "$url")" "$url"; \
    done; \
    sha256sum -c SHA256SUMS; \
    for f in *.tar.*; do tar xf "$f"; done

# LIBRARY_PATH lets meson's find_library() see libraries installed earlier in
# this stage (pangomm needs glibmm_generate_extra_defs from glibmm).
ENV PKG_CONFIG_PATH=/opt/inkscape/lib/pkgconfig:/opt/libvisio/lib/pkgconfig \
    LD_LIBRARY_PATH=/opt/inkscape/lib:/opt/libvisio/lib \
    LIBRARY_PATH=/opt/inkscape/lib:/opt/libvisio/lib

# gtkmm-3 binding stack (meson builds, docs/examples off).
RUN set -eu; \
    for d in libsigc++-2.12.1 glibmm-2.66.10 cairomm-1.14.6 pangomm-2.46.5 atkmm-2.28.5 gtkmm-3.24.11; do \
      meson setup "build-$d" "$d" --prefix=/opt/inkscape --libdir=lib --buildtype=release \
        -Dbuild-documentation=false -Dmaintainer-mode=false \
        $(case "$d" in libsigc*|glibmm*|cairomm*) echo -Dbuild-examples=false;; esac) \
        $(case "$d" in cairomm*|libsigc*) echo -Dbuild-tests=false;; esac) \
        $(case "$d" in gtkmm*) echo -Dbuild-demos=false -Dbuild-tests=false;; esac); \
      ninja -C "build-$d" -j "${BUILD_JOBS:-$(nproc)}"; \
      ninja -C "build-$d" install; \
    done

# Inkscape: CLI export only needs the core; optional importers and GUI extras off.
RUN set -eu; \
    src="$(ls -d inkscape-1.4.4*/)"; \
    cmake -S "$src" -B build-inkscape -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/inkscape \
      -DCMAKE_INSTALL_RPATH='/opt/inkscape/lib;/opt/inkscape/lib/inkscape' \
      -DBUILD_TESTING=OFF -DWITH_NLS=OFF -DWITH_MANPAGE_COMPILE=OFF \
      -DENABLE_POPPLER=OFF -DENABLE_POPPLER_CAIRO=OFF \
      -DWITH_GSPELL=OFF -DWITH_GSOURCEVIEW=OFF -DWITH_DBUS=OFF \
      -DWITH_IMAGE_MAGICK=OFF -DWITH_GRAPHICS_MAGICK=OFF \
      -DWITH_LIBCDR=OFF -DWITH_LIBVISIO=OFF -DWITH_LIBWPG=OFF \
      -DWITH_JEMALLOC=OFF -DWITH_OPENMP=OFF -DWITH_X11=OFF; \
    ninja -C build-inkscape -j "${BUILD_JOBS:-$(nproc)}"; \
    ninja -C build-inkscape install; \
    rm -rf /opt/inkscape/share/inkscape/extensions /opt/inkscape/share/man \
           /opt/inkscape/share/doc /opt/inkscape/include /opt/inkscape/share/aclocal

# librevenge + libvisio with the vsd2xhtml converter.
RUN set -eu; \
    cd /src/librevenge-0.0.5; \
    ./configure --prefix=/opt/libvisio --disable-static --disable-werror \
      --without-docs --disable-tests; \
    make -j "${BUILD_JOBS:-$(nproc)}"; make install; \
    cd /src/libvisio-0.1.8; \
    # ICU 7x headers need C++17; libvisio's configure would otherwise pin C++11.
    ./configure CXX='g++ -std=c++17' --prefix=/opt/libvisio --disable-static --disable-werror \
      --without-docs --disable-tests; \
    make -j "${BUILD_JOBS:-$(nproc)}"; make install; \
    rm -rf /opt/libvisio/include /opt/libvisio/share/doc

# DejaVu (Debian's default sans-serif) keeps Visio text measurement and
# rendered output identical to the previous Debian image. Wolfi has no package.
RUN set -eu; \
    mkdir -p /opt/fonts/dejavu; \
    cp /src/dejavu-fonts-ttf-2.37/ttf/*.ttf /opt/fonts/dejavu/; \
    cp /src/dejavu-fonts-ttf-2.37/LICENSE /opt/fonts/dejavu/

# Verify every copied binary/library resolves, then record the system sonames
# the runtime stage must install (apk `so:` provides). Wolfi has no ldd; call
# the dynamic loader directly (that is all ldd does).
RUN set -eu; \
    LD=/usr/lib/ld-linux-x86-64.so.2; \
    /opt/inkscape/bin/inkscape --version; \
    test -x /opt/libvisio/bin/vsd2xhtml; \
    find /opt/inkscape /opt/libvisio -type f \( -perm -u+x -o -name '*.so*' \) > /tmp/objects; \
    while read -r f; do "$LD" --list "$f" 2>/dev/null || true; done < /tmp/objects > /tmp/resolved; \
    if grep -q 'not found' /tmp/resolved; then grep 'not found' /tmp/resolved | sort -u; exit 1; fi; \
    test "$(grep -c '=> /' /tmp/resolved)" -gt 50; \
    awk '/=> \/usr\/lib\//{print "so:" $1}' /tmp/resolved | grep -vE '^so:(/|ld-linux)' | sort -u > /opt/native-runtime-deps.txt; \
    wc -l /opt/native-runtime-deps.txt


# Stage 1: Build dependencies
FROM ${WOLFI_IMAGE} AS builder
WORKDIR /app

# ca-certificates needed so optional custom roots can be merged into the
# system trust store before pip hits HTTPS (public or internal).
# build-base/python-dev and the -dev libraries build pikepdf and OpenCV from
# source against Wolfi's patched libraries (see docker/source-built-wheels.txt);
# builder-only.
RUN apk add --no-cache python-3.13 python-3.13-dev build-base bash ca-certificates \
      cmake samurai pkgconf qpdf-dev libjpeg-turbo-dev libpng-dev tiff-dev \
      libwebp-dev zlib-dev

# Optional custom roots: drop certs/Trusted_Root_CAs.pem in the build context.
# Primary use case: corporate MITM / TLS inspection proxies that re-sign
# outbound HTTPS (Zscaler, Palo Alto, Netskope, Blue Coat, etc.). Without
# that proxy CA in the trust store, pip fails with CERTIFICATE_VERIFY_FAILED
# even when the network is not air-gapped.
# Directory always exists (certs/.gitkeep); the .pem itself is optional.
# Run via `sh` (not ./script) so CRLF checkouts / missing +x never produce a
# cryptic "/tmp/install_trusted_root_cas.sh: not found" from a broken shebang.
COPY certs/ /tmp/certs/
COPY scripts/install_trusted_root_cas.sh /tmp/install_trusted_root_cas.sh
RUN sed -i 's/\r$//' /tmp/install_trusted_root_cas.sh \
 && echo "certs/ contents:" && ls -la /tmp/certs/ \
 && sh /tmp/install_trusted_root_cas.sh /tmp/certs/Trusted_Root_CAs.pem

# Prefer the system bundle (includes any custom roots) over certifi alone.
# PIP_CERT is required: pip does not always honor SSL_CERT_FILE for index TLS.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Explicit forward proxy (only if inspection is NOT transparent).
# Docker also accepts the usual --build-arg HTTP_PROXY=... from the client.
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY=
ARG http_proxy=
ARG https_proxy=
ARG no_proxy=
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# Package indexes (optional overrides for air-gap mirrors).
ARG PIP_INDEX_URL=
ARG PIP_EXTRA_INDEX_URL=
# MITM proxies often break pip TLS even when a custom CA is installed (pip's
# certifi path, incomplete chain, etc.). Default trusted-host list covers the
# public indexes used by this Dockerfile; override/extend via build-arg.
# Space-separated hostnames, e.g. "pypi.org files.pythonhosted.org my-pypi.local"
ARG PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org pythonhosted.org download.pytorch.org"
# Persist pip config for all subsequent pip invocations (incl. venv).
# Note: ConfigParser forbids repeated keys — use one trusted-host with a
# multi-line indented list (pip's documented form), not multiple assignments.
RUN set -eu; \
    { \
      echo "[global]"; \
      echo "cert = /etc/ssl/certs/ca-certificates.crt"; \
      if [ -n "${PIP_INDEX_URL}" ]; then echo "index-url = ${PIP_INDEX_URL}"; fi; \
      if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then echo "extra-index-url = ${PIP_EXTRA_INDEX_URL}"; fi; \
      if [ -n "${PIP_TRUSTED_HOST}" ]; then \
        echo "trusted-host ="; \
        for h in ${PIP_TRUSTED_HOST}; do echo "    ${h}"; done; \
      fi; \
    } > /etc/pip.conf; \
    mkdir -p /etc/xdg/pip && cp /etc/pip.conf /etc/xdg/pip/pip.conf; \
    echo "---- /etc/pip.conf ----"; cat /etc/pip.conf; echo "-----------------------"

# Isolated venv so CPU torch is visible to the second pip install ( --prefix
# installs are not considered "installed" by a later bare pip resolve).
RUN python3.13 -m venv /opt/venv \
 && mkdir -p /opt/venv/pip \
 && cp /etc/pip.conf /opt/venv/pip/pip.conf
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_CONFIG_FILE=/etc/pip.conf

# pikepdf and OpenCV from source. Their PyPI wheels bundle outdated native
# libraries that scanners cannot see (pikepdf: libjpeg-turbo 1.5.3; OpenCV:
# OpenSSL 1.1.1k, FFmpeg, Qt 5, X11, OpenBLAS 0.3.15). These builds link Wolfi's
# qpdf/libjpeg-turbo/libpng/libtiff/libwebp/zlib instead (no JPEG 2000 in
# OpenCV; Pillow decodes images, OpenCV only processes arrays). OpenCV is
# limited to the modules Sauron's dependencies use (core, imgproc, imgcodecs,
# videoio without any capture backend; features/flann/geometry/calib only
# because OpenCV's Python typing-stub generator references them). No DNN
# (bundled protobuf), GUI, IPP or other third-party downloads.
COPY docker/source-built-wheels.txt /tmp/source-built-wheels.txt
RUN set -eu; \
    export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" MAKEFLAGS="-j$(nproc)" ENABLE_HEADLESS=1; \
    export CMAKE_ARGS="-DBUILD_LIST=core,imgproc,imgcodecs,videoio,features,flann,geometry,calib,python3 \
      -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF -DWITH_1394=OFF \
      -DWITH_IPP=OFF -DWITH_ADE=OFF -DWITH_PROTOBUF=OFF -DWITH_OPENEXR=OFF \
      -DWITH_JASPER=OFF -DWITH_OPENJPEG=OFF -DWITH_AVIF=OFF -DWITH_OPENCL=OFF -DWITH_LAPACK=OFF \
      -DWITH_EIGEN=OFF -DWITH_QT=OFF -DWITH_GTK=OFF \
      -DBUILD_JPEG=OFF -DBUILD_PNG=OFF -DBUILD_TIFF=OFF -DBUILD_WEBP=OFF \
      -DBUILD_OPENJPEG=OFF -DBUILD_ZLIB=OFF -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF"; \
    pip wheel --no-cache-dir --no-deps --no-binary pikepdf,opencv-python-headless \
      --require-hashes -r /tmp/source-built-wheels.txt -w /opt/source-wheels; \
    ls -l /opt/source-wheels

COPY requirements.txt constraints-security.txt ./

# CPU-only PyTorch. Default PyPI Linux wheels pull multi-GB nvidia-* CUDA
# packages we never need here (local embeddings + CrossEncoder run on CPU;
# LLM inference is external). Install from the official CPU wheel index first;
# with torch already satisfied, the full requirements install will not replace
# it with a CUDA build from PyPI.
#
# Override only if you mirror torch (air-gap). MITM sites can leave the default
# once certs/Trusted_Root_CAs.pem trusts the inspection proxy.
# Re-declare ARG so this RUN layer sees the values (Docker ARG scope).
ARG PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org pythonhosted.org download.pytorch.org"
ARG PIP_INDEX_URL=
ARG PIP_EXTRA_INDEX_URL=
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
RUN set -eu; \
    TH_ARGS=""; \
    for h in ${PIP_TRUSTED_HOST}; do TH_ARGS="${TH_ARGS} --trusted-host ${h}"; done; \
    IDX_ARGS=""; \
    if [ -n "${PIP_INDEX_URL}" ]; then IDX_ARGS="${IDX_ARGS} -i ${PIP_INDEX_URL}"; fi; \
    if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then IDX_ARGS="${IDX_ARGS} --extra-index-url ${PIP_EXTRA_INDEX_URL}"; fi; \
    echo "pip trusted-host args:${TH_ARGS}"; \
    echo "pip index args:${IDX_ARGS}"; \
    pip install --no-cache-dir --upgrade 'pip>=26.2.0' 'setuptools>=83.0.0' 'wheel>=0.46.2' \
      -c constraints-security.txt \
      --cert /etc/ssl/certs/ca-certificates.crt \
      ${TH_ARGS} ${IDX_ARGS}; \
    pip install --no-cache-dir \
      torch torchvision \
      -c constraints-security.txt \
      --index-url "${TORCH_CPU_INDEX}" \
      --cert /etc/ssl/certs/ca-certificates.crt \
      ${TH_ARGS}; \
    pip install --no-cache-dir /opt/source-wheels/*.whl \
      -c constraints-security.txt \
      --cert /etc/ssl/certs/ca-certificates.crt \
      ${TH_ARGS} ${IDX_ARGS}; \
    pip install --no-cache-dir \
      -r requirements.txt \
      -c constraints-security.txt \
      --cert /etc/ssl/certs/ca-certificates.crt \
      ${TH_ARGS} ${IDX_ARGS}; \
    # unstructured-inference depends on opencv-python by name; replace that
    # bundled-library wheel with the source-built headless OpenCV (same cv2).
    pip uninstall -y opencv-python opencv-contrib-python >/dev/null 2>&1 || true; \
    pip install --no-cache-dir --no-deps --force-reinstall /opt/source-wheels/opencv_python_headless-*.whl; \
    pip uninstall -y hf-xet hf_xet >/dev/null 2>&1 || true; \
    python - <<'PY'
import importlib.metadata as md
import pathlib
import torch

print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
site = pathlib.Path(torch.__file__).resolve().parents[1]
if site.name != "site-packages":
    site = site.parent
bad = sorted(
    p.name
    for p in site.iterdir()
    if p.name.startswith(("nvidia", "cuda_")) or p.name in ("cuda", "nvidia")
)
assert not bad, f"CUDA/NVIDIA packages leaked into image: {bad}"
# Local version tag from the CPU wheel index (e.g. 2.13.0+cpu)
assert "+cpu" in torch.__version__ or not torch.cuda.is_available(), (
    f"unexpected torch build: {torch.__version__}"
)
print("OK: CPU-only torch (no nvidia-* packages)")

# Source-built wheels are the ones installed, with no bundled native copies.
import cv2
import pikepdf
for name in ("opencv-python", "opencv-contrib-python"):
    try:
        md.distribution(name)
    except md.PackageNotFoundError:
        continue
    raise AssertionError(f"{name} (bundled FFmpeg/OpenSSL/Qt) is installed")
for libs in ("opencv_python.libs", "opencv_python_headless.libs", "pikepdf.libs"):
    assert not (site / libs).exists(), f"bundled native libraries present: {libs}"
info = cv2.getBuildInformation()
video_io = [l for l in info.splitlines() if l.strip().startswith(("FFMPEG:", "GStreamer:", "v4l/v4l2:"))]
assert all(l.rstrip().endswith("NO") for l in video_io), video_io
print(f"OK: cv2 {cv2.__version__} (no video backends), pikepdf {pikepdf.__version__} qpdf {pikepdf.__libqpdf_version__}")
PY

# Record the Wolfi packages owning the shared libraries the venv's extensions
# link against (the source-built wheels link system libraries); runtime-base
# installs exactly those packages.
RUN set -eu; \
    LD=/usr/lib/ld-linux-x86-64.so.2; \
    find /opt/venv -type f -name '*.so*' | while read -r f; do "$LD" --list "$f" 2>/dev/null || true; done \
      | awk '/=> \/usr\/lib\//{print $3}' | sort -u | while read -r lib; do \
          apk info -q --who-owns "$(readlink -f "$lib")" 2>/dev/null | sed -E 's/.* is owned by //'; \
        done | sed -E 's/-[0-9][^-]*-r[0-9]+$//' | grep -vE '^(glibc|ld-linux)' | sort -u > /opt/venv-runtime-deps.txt; \
    cat /opt/venv-runtime-deps.txt | tr '\n' ' '; echo

# Merge OS CA bundle (incl. MITM roots) into certifi so huggingface_hub /
# requests / urllib3 trust the same roots as the system store.
COPY scripts/inject_system_cas_into_certifi.py /tmp/inject_system_cas_into_certifi.py
RUN python /tmp/inject_system_cas_into_certifi.py

# pip is only needed to build the venv. Its vendored dependency manifest
# (pip/_vendor/vendor.txt: msgpack 1.1.2, setuptools 70.3.0) is reported by
# scanners even in the newest pip, and the runtime never installs packages.
RUN python -m pip uninstall -y pip \
 && ! python -c 'import pip' 2>/dev/null

# A normal `COPY --from=builder /opt/venv /opt/venv` collapses the complete
# 3+ GiB environment into one image layer. Partition it into deterministic
# overlay trees; the final stage copies each tree as its own bounded layer.
COPY scripts/split_layer_tree.py /tmp/split_layer_tree.py
RUN python /tmp/split_layer_tree.py \
      --source /opt/venv \
      --output /opt/venv-layers \
      --layer-count 16 \
      --max-bytes 850000000

# Stage 2: Runtime
FROM ${WOLFI_IMAGE} AS runtime-base
WORKDIR /app

# System dependencies for document parsing.
# Shared libraries needed by the Inkscape/libvisio builds (native-tools stage)
# and by the source-built OpenCV/pikepdf wheels (builder stage) are installed
# below from the sonames those stages recorded; the runtime stage then fails
# the build if any shipped shared object has an unresolved library.
# No curl: the health check uses Python's stdlib.
RUN apk upgrade --no-cache && apk add --no-cache \
      python-3.13 bash ca-certificates \
      tesseract tesseract-eng libmagic poppler-utils rsvg-convert \
      fontconfig font-liberation libstdc++

# Inkscape (EMF rendering), vsd2xhtml (Visio rendering) and DejaVu fonts,
# built/pinned in the native-tools stage, plus the shared libraries they link
# (recorded as apk `so:` names at build time so they track Wolfi versions).
COPY --from=native-tools /opt/native-runtime-deps.txt /tmp/native-runtime-deps.txt
COPY --from=builder /opt/venv-runtime-deps.txt /tmp/venv-runtime-deps.txt
RUN apk add --no-cache $(cat /tmp/native-runtime-deps.txt /tmp/venv-runtime-deps.txt | sort -u) \
 && rm /tmp/native-runtime-deps.txt /tmp/venv-runtime-deps.txt
COPY --from=native-tools /opt/inkscape /opt/inkscape
COPY --from=native-tools /opt/libvisio /opt/libvisio
COPY --from=native-tools /opt/fonts/dejavu /usr/share/fonts/dejavu
RUN set -eu; \
    printf '%s\n' /opt/inkscape/lib /opt/inkscape/lib64 /opt/inkscape/lib64/inkscape \
      /opt/libvisio/lib >> /etc/ld.so.conf; \
    ldconfig; \
    ln -s /opt/inkscape/bin/inkscape /usr/bin/inkscape; \
    for tool in /opt/libvisio/bin/*; do ln -s "$tool" /usr/bin/; done; \
    fc-cache -f >/dev/null; \
    inkscape --version; command -v vsd2xhtml rsvg-convert pdftoppm tesseract fc-match

# The application runs from /opt/venv (secured pip/setuptools live there).
# Remove the OS Python's bundled tooling, which the runtime never uses.
RUN set -eu; \
    rm -rf /usr/lib/python3.13/site-packages/pip* \
           /usr/lib/python3.13/site-packages/setuptools* \
           /usr/lib/python3.13/site-packages/_distutils_hack \
           /usr/lib/python3.13/site-packages/distutils-precedence.pth \
           /usr/lib/python3.13/site-packages/pkg_resources \
           /usr/lib/python3.13/site-packages/wheel* \
           /usr/lib/python3.13/ensurepip/_bundled/*.whl \
           /usr/bin/pip /usr/bin/pip3 /usr/bin/pip3.13; \
    ! python3 -c 'import pip' 2>/dev/null

# Same optional custom roots as the builder (outbound LLM/embed HTTPS, etc.).
COPY certs/ /tmp/certs/
COPY scripts/install_trusted_root_cas.sh /tmp/install_trusted_root_cas.sh
RUN sed -i 's/\r$//' /tmp/install_trusted_root_cas.sh \
 && echo "certs/ contents:" && ls -la /tmp/certs/ \
 && sh /tmp/install_trusted_root_cas.sh /tmp/certs/Trusted_Root_CAs.pem \
 && rm -rf /tmp/certs /tmp/install_trusted_root_cas.sh

# Prefer the system bundle (includes any custom roots) over certifi alone.
# Same MITM CA trust as builder so runtime LLM/embed HTTPS through the proxy works.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

# Explicit proxy for runtime layer too (prefetch + later LLM calls if set).
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG NO_PROXY=
ARG http_proxy=
ARG https_proxy=
ARG no_proxy=
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
RUN chmod +x scripts/entrypoint.sh scripts/inject_system_cas_into_certifi.py \
    scripts/split_layer_tree.py scripts/check_oci_layer_sizes.py

# Stage 3: Download and validate offline assets. Large caches stay in this
# intermediate stage; only the bounded overlay trees are copied into runtime.
FROM runtime-base AS model-builder

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Re-merge system CAs into certifi and explicitly configure Hugging Face Hub
# below. The builder and runtime stages can have different OS CA bundles.
RUN python scripts/inject_system_cas_into_certifi.py

# Bake ALL Hugging Face / local ML weights required at runtime:
#   nomic embeddings, cross-encoder rerankers, YOLOX layout, table-transformer.
# Default: hard-fail the build if any download fails (offline-ready image).
#
#   --build-arg SAURON_PREFETCH_INSECURE_SSL=1   # diagnostic only; prefer CA PEM
#   --build-arg SAURON_PREFETCH_ALLOW_FAIL=1     # do not fail build (not recommended)
#   --build-arg SKIP_HF_MODEL_PREFETCH=1         # skip bake (runtime needs HF)
# Optional: pre-seed host cache into the image (air-gap friendly):
#   mkdir -p hf-cache && # copy ~/.cache/huggingface contents here
# Optional host cache is copied when present (see COPY below).
ARG SAURON_PREFETCH_INSECURE_SSL=0
ARG SKIP_PDF_MODEL_PREFETCH=0
ARG SKIP_HF_MODEL_PREFETCH=0
ARG SAURON_PREFETCH_ALLOW_FAIL=0
ARG EMBEDDING_MODEL_NAME=nomic-ai/nomic-embed-text-v1
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# Hugging Face hub (public or Artifactory / corporate mirror). Used only during
# model bake — not left as permanent runtime env for the token.
# Example Artifactory remote:
#   HF_ENDPOINT=https://artifactory.example.com/artifactory/api/huggingfaceml/huggingface-remote
ARG HF_ENDPOINT=
# Optional Hugging Face token for the model download only: pass it as a
# BuildKit secret (id=hf_token), never a build arg, so it is not recorded in
# image history, provenance or layers. Not needed at runtime (models are baked).
ENV SAURON_PREFETCH_INSECURE_SSL=${SAURON_PREFETCH_INSECURE_SSL} \
    SKIP_PDF_MODEL_PREFETCH=${SKIP_PDF_MODEL_PREFETCH} \
    SKIP_HF_MODEL_PREFETCH=${SKIP_HF_MODEL_PREFETCH} \
    SAURON_PREFETCH_ALLOW_FAIL=${SAURON_PREFETCH_ALLOW_FAIL} \
    EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME} \
    RERANK_MODEL=${RERANK_MODEL} \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HOME=/opt/models/huggingface \
    TRANSFORMERS_CACHE=/opt/models/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/opt/models/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence_transformers \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken

# Optional pre-seeded HF hub cache from build context (for true air-gap builds).
# Create hf-cache/ on the host with hub/ blobs from a machine that can reach HF.
COPY hf-cache/ /opt/models/huggingface/

# Optional pre-seeded tiktoken cache for builds that cannot reach OpenAI blob
# storage. On connected builds scripts/prefetch_hf_models.py fills this path.
COPY tiktoken-cache/ /app/.cache/tiktoken/

COPY tests/fixtures/pdf/tiny_smoke.pdf tests/fixtures/pdf/tiny_smoke.pdf
# HF_ENDPOINT and the hf_token secret apply only to this RUN (bake). Prefer an
# Artifactory remote URL when public huggingface.co is blocked or slow.
# Do not export empty HF_ENDPOINT= — hub treats that as a blank base URL and fails.
# Cache partitioning stays in this RUN so the export trees can hard-link the
# freshly downloaded files instead of copying them in the intermediate stage.
RUN --mount=type=secret,id=hf_token \
    set -eu; \
    echo "build SAURON_PREFETCH_INSECURE_SSL=${SAURON_PREFETCH_INSECURE_SSL} ALLOW_FAIL=${SAURON_PREFETCH_ALLOW_FAIL} SKIP=${SKIP_HF_MODEL_PREFETCH} HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co (default)}"; \
    export SAURON_PREFETCH_INSECURE_SSL="${SAURON_PREFETCH_INSECURE_SSL}" \
      SKIP_PDF_MODEL_PREFETCH="${SKIP_PDF_MODEL_PREFETCH}" \
      SKIP_HF_MODEL_PREFETCH="${SKIP_HF_MODEL_PREFETCH}" \
      SAURON_PREFETCH_ALLOW_FAIL="${SAURON_PREFETCH_ALLOW_FAIL}" \
      EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME}" \
      RERANK_MODEL="${RERANK_MODEL}" \
      HF_HUB_DISABLE_XET=1 \
      HF_HUB_ENABLE_HF_TRANSFER=0; \
    if [ -n "${HF_ENDPOINT}" ]; then export HF_ENDPOINT="${HF_ENDPOINT}"; else unset HF_ENDPOINT || true; fi; \
    if [ -s /run/secrets/hf_token ]; then \
      HF_TOKEN="$(cat /run/secrets/hf_token)"; HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"; \
      export HF_TOKEN HUGGING_FACE_HUB_TOKEN; echo "hf_token secret: provided"; \
    else unset HF_TOKEN HUGGING_FACE_HUB_TOKEN || true; fi; \
    python scripts/prefetch_hf_models.py; \
    if [ "${SAURON_PREFETCH_ALLOW_FAIL}" != "1" ] && [ "${SKIP_HF_MODEL_PREFETCH}" != "1" ] && [ "${SKIP_PDF_MODEL_PREFETCH}" != "1" ]; then \
      test -f /app/.pdf_models_ready; \
    fi; \
    mkdir -p /opt/model-export/opt /opt/model-export/app; \
    if [ -d /opt/models ]; then cp -al /opt/models /opt/model-export/opt/; fi; \
    if [ -d /app/.cache ]; then cp -al /app/.cache /opt/model-export/app/; fi; \
    for marker in /app/.pdf_models_ready /app/.pdf_models_prefetch_failed; do \
      if [ -f "${marker}" ]; then cp -a "${marker}" /opt/model-export/app/; fi; \
    done; \
    python scripts/split_layer_tree.py \
      --source /opt/model-export \
      --output /opt/model-layers \
      --layer-count 16 \
      --max-bytes 850000000

# Stage 4: Production image. Each COPY below becomes a separate OCI layer.
# The splitter leaves 15% headroom below the CI's decimal 1 GB compressed-blob
# limit for tar metadata and future package/model growth.
FROM runtime-base AS runtime

COPY --from=builder /opt/venv-layers/00/ /opt/venv/
COPY --from=builder /opt/venv-layers/01/ /opt/venv/
COPY --from=builder /opt/venv-layers/02/ /opt/venv/
COPY --from=builder /opt/venv-layers/03/ /opt/venv/
COPY --from=builder /opt/venv-layers/04/ /opt/venv/
COPY --from=builder /opt/venv-layers/05/ /opt/venv/
COPY --from=builder /opt/venv-layers/06/ /opt/venv/
COPY --from=builder /opt/venv-layers/07/ /opt/venv/
COPY --from=builder /opt/venv-layers/08/ /opt/venv/
COPY --from=builder /opt/venv-layers/09/ /opt/venv/
COPY --from=builder /opt/venv-layers/10/ /opt/venv/
COPY --from=builder /opt/venv-layers/11/ /opt/venv/
COPY --from=builder /opt/venv-layers/12/ /opt/venv/
COPY --from=builder /opt/venv-layers/13/ /opt/venv/
COPY --from=builder /opt/venv-layers/14/ /opt/venv/
COPY --from=builder /opt/venv-layers/15/ /opt/venv/
ENV PATH="/opt/venv/bin:$PATH"

RUN python scripts/inject_system_cas_into_certifi.py

# Every shared object shipped must resolve against the runtime's libraries.
RUN set -eu; \
    LD=/usr/lib/ld-linux-x86-64.so.2; \
    find /opt/venv /opt/inkscape /opt/libvisio -type f -name '*.so*' | while read -r f; do \
      "$LD" --list "$f" 2>/dev/null | grep 'not found' | sed "s#^#${f}: #" || true; \
    done > /tmp/unresolved; \
    if [ -s /tmp/unresolved ]; then sort -u /tmp/unresolved | head -50; exit 1; fi

COPY --from=model-builder /opt/model-layers/00/ /
COPY --from=model-builder /opt/model-layers/01/ /
COPY --from=model-builder /opt/model-layers/02/ /
COPY --from=model-builder /opt/model-layers/03/ /
COPY --from=model-builder /opt/model-layers/04/ /
COPY --from=model-builder /opt/model-layers/05/ /
COPY --from=model-builder /opt/model-layers/06/ /
COPY --from=model-builder /opt/model-layers/07/ /
COPY --from=model-builder /opt/model-layers/08/ /
COPY --from=model-builder /opt/model-layers/09/ /
COPY --from=model-builder /opt/model-layers/10/ /
COPY --from=model-builder /opt/model-layers/11/ /
COPY --from=model-builder /opt/model-layers/12/ /
COPY --from=model-builder /opt/model-layers/13/ /
COPY --from=model-builder /opt/model-layers/14/ /
COPY --from=model-builder /opt/model-layers/15/ /

# Keep the operational parser smoke fixture available in the production image.
COPY tests/fixtures/pdf/tiny_smoke.pdf tests/fixtures/pdf/tiny_smoke.pdf

ARG EMBEDDING_MODEL_NAME=nomic-ai/nomic-embed-text-v1
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Create data directory and prove the selected prefetch policy left a marker.
# Run as Wolfi's standard unprivileged user. Models, code and caches stay
# root-owned and read-only; only /app/data (the volume) and the user's home
# (library scratch such as matplotlib's font cache) are writable.
# New named volumes inherit this ownership; existing root-owned volumes are
# re-owned once by the compose `data-permissions` service (or Kubernetes
# fsGroup).
ARG APP_UID=65532
ARG APP_GID=65532
RUN set -eu; \
    mkdir -p /app/data/lancedb /home/nonroot; \
    chown -R "${APP_UID}:${APP_GID}" /app/data /home/nonroot; \
    test -f /app/.pdf_models_ready -o -f /app/.pdf_models_prefetch_failed

# HF cache paths + offline by default (models baked above). Entrypoint reinforces
# offline when /app/.pdf_models_ready exists.
ENV LANCEDB_PATH=/app/data/lancedb \
    LANCEDB_TABLE_NAME=chunks \
    DATABASE_URL=sqlite+aiosqlite:///./data/metadata.db \
    VLLM_REQUEST_TIMEOUT=300 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME} \
    RERANK_MODEL=${RERANK_MODEL} \
    HF_HOME=/opt/models/huggingface \
    TRANSFORMERS_CACHE=/opt/models/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/opt/models/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/sentence_transformers \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    HOME=/home/nonroot

EXPOSE 8080
VOLUME /app/data

# Probe the public sign-in page; /api/health requires an application API key.
# Python stdlib instead of curl keeps the curl CLI out of the runtime image.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/admin/login', timeout=4).status == 200 else 1)"]

USER 65532:65532

ENTRYPOINT ["scripts/entrypoint.sh"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
