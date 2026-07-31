#!/bin/sh
# Install an optional PEM bundle of custom root CAs into the OS trust store.
# Prefer: sh install_trusted_root_cas.sh  (avoids shebang/CRLF exec issues)
#
# Usage: install_trusted_root_cas.sh [path-to-Trusted_Root_CAs.pem]
# If the file is missing or empty, exit 0 and leave the default trust store.
#
# After this runs, point Python/HTTP clients at the system bundle:
#   SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
#   REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
#   CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
#   PIP_CERT=/etc/ssl/certs/ca-certificates.crt
set -eu

PEM="${1:-/tmp/certs/Trusted_Root_CAs.pem}"
DEST_DIR="/usr/local/share/ca-certificates"
DEST_CRT="${DEST_DIR}/Trusted_Root_CAs.crt"

if [ ! -f "$PEM" ]; then
  echo "install_trusted_root_cas: no file at ${PEM}; using default CA trust store"
  echo "install_trusted_root_cas: if pip TLS fails (MITM inspection / private CA), add certs/Trusted_Root_CAs.pem"
  exit 0
fi

if [ ! -s "$PEM" ]; then
  echo "install_trusted_root_cas: ${PEM} is empty; using default CA trust store"
  exit 0
fi

if ! command -v update-ca-certificates >/dev/null 2>&1; then
  echo "install_trusted_root_cas: update-ca-certificates not found; install ca-certificates package" >&2
  exit 1
fi

# Normalize CRLF in the PEM (common when files are copied from Windows).
sed -i 's/\r$//' "$PEM" 2>/dev/null || sed -i '' 's/\r$//' "$PEM" 2>/dev/null || true

if ! grep -q "BEGIN CERTIFICATE" "$PEM"; then
  echo "install_trusted_root_cas: ${PEM} has no BEGIN CERTIFICATE markers (need PEM, not DER/JKS)" >&2
  exit 1
fi

ncerts=$(grep -c "BEGIN CERTIFICATE" "$PEM" || true)
echo "install_trusted_root_cas: found ${ncerts} certificate(s) in ${PEM}"

mkdir -p "$DEST_DIR"
# Debian expects .crt under /usr/local/share/ca-certificates/ (PEM content is fine).
cp "$PEM" "$DEST_CRT"
chmod 644 "$DEST_CRT"
update-ca-certificates

echo "install_trusted_root_cas: installed custom roots from ${PEM}"
# Quick sanity: system bundle should grow / still exist
if [ ! -s /etc/ssl/certs/ca-certificates.crt ]; then
  echo "install_trusted_root_cas: /etc/ssl/certs/ca-certificates.crt missing after update" >&2
  exit 1
fi
