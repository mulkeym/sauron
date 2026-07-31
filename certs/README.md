# Custom root CAs (optional)

Place a PEM bundle at:

```text
certs/Trusted_Root_CAs.pem
```

If that file is present at **Docker build** time, the image installs it into the
system trust store and points Python / requests / curl at that store. If the
file is absent, the build uses the default public CA set only.

The PEM may contain one or more certificates (concatenated).

```bash
# Example: drop your internal roots, then build
cp /path/to/org-roots.pem certs/Trusted_Root_CAs.pem
docker build -t sauron .
```

`Trusted_Root_CAs.pem` is gitignored so environment-specific CAs are not
committed by accident. Force-add it if you intentionally want it in the repo.

## Air-gapped / private HTTPS package indexes

If `pip install` fails with SSL errors during the Docker build:

1. Confirm the build log shows your PEM was installed (not
   `no file at .../Trusted_Root_CAs.pem`).
2. Put the **root (or issuing) CA that signs your internal PyPI / proxy** in
   `certs/Trusted_Root_CAs.pem` (PEM, one or more certs concatenated).
3. Point pip at your mirror and, if needed, torch at the same place:

```bash
docker build -t sauron \
  --build-arg PIP_INDEX_URL=https://pypi.internal.example/simple \
  --build-arg TORCH_CPU_INDEX=https://pypi.internal.example/simple \
  .
```

Optional escape hatch when you cannot load the CA yet (TLS verification
skipped for those hosts only — prefer fixing the PEM):

```bash
docker build -t sauron \
  --build-arg PIP_INDEX_URL=https://pypi.internal.example/simple \
  --build-arg PIP_TRUSTED_HOST="pypi.internal.example files.internal.example" \
  --build-arg TORCH_CPU_INDEX=https://pypi.internal.example/simple \
  .
```

The Dockerfile sets `PIP_CERT` / `SSL_CERT_FILE` to the system CA bundle so
pip uses the same trust store as the OS after your roots are installed.
