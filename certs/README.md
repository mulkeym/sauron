# Custom root CAs (optional)

Place a PEM bundle at:

```text
certs/Trusted_Root_CAs.pem
```

If that file is present at **Docker build** time, the image installs it into the
system trust store and points Python / requests / curl / pip at that store. If
the file is absent, the build uses the default public CA set only.

The PEM may contain one or more certificates (concatenated).

## Corporate MITM / TLS inspection (most common)

If outbound HTTPS is intercepted (Zscaler, Palo Alto, Netskope, Blue Coat,
etc.), every site cert is **re-signed by the proxy**. Clients that only trust
public CAs then fail with `CERTIFICATE_VERIFY_FAILED` — including `pip` during
`docker build` — even when you are **not** air-gapped.

**Fix:** put the **TLS inspection / SSL decrypt root (or issuing) CA** in
`certs/Trusted_Root_CAs.pem`, then build normally:

```bash
# Export the inspection CA from your security team / browser trust store
cp /path/to/corp-inspection-ca.pem certs/Trusted_Root_CAs.pem
docker build -t sauron .
```

You do **not** need a private PyPI mirror for this case. Default PyPI and
`download.pytorch.org` work once the proxy CA is trusted.

Confirm in the build log:

```text
certs/ contents:
... Trusted_Root_CAs.pem ...
install_trusted_root_cas: found N certificate(s) ...
install_trusted_root_cas: installed custom roots from ...
```

If you see `no file at .../Trusted_Root_CAs.pem`, the CA never entered the
build context.

### Explicit proxy (non-transparent)

If clients must set proxy env vars (not pure transparent intercept):

```bash
docker build -t sauron \
  --build-arg HTTP_PROXY=http://proxy.example:8080 \
  --build-arg HTTPS_PROXY=http://proxy.example:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  .
```

Still include the inspection CA in `Trusted_Root_CAs.pem`.

## Air-gapped / private package indexes

If you **also** host an internal PyPI (no internet):

```bash
docker build -t sauron \
  --build-arg PIP_INDEX_URL=https://pypi.internal.example/simple \
  --build-arg TORCH_CPU_INDEX=https://pypi.internal.example/simple \
  .
```

### pip `--trusted-host` (MITM escape hatch)

The Dockerfile **defaults** `PIP_TRUSTED_HOST` to the public indexes used at
build time (`pypi.org`, `files.pythonhosted.org`, `pypi.python.org`,
`download.pytorch.org`) and passes `--trusted-host` on every builder `pip`
call. That skips TLS verification for those hosts when inspection still
breaks verification even with the CA PEM installed.

Extend or replace the list if you use other hosts:

```bash
docker build -t sauron \
  --build-arg PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org download.pytorch.org pypi.internal.example" \
  .
```

To disable trusted-host entirely (strict TLS only):

```bash
docker build -t sauron --build-arg PIP_TRUSTED_HOST="" .
```

## Runtime + Hugging Face model prefetch

The same PEM is installed in the **runtime** image stage so LLM / embedding
HTTPS through the MITM proxy trusts the inspection CA. Prefer this over
**Admin → Models → Ignore SSL certificate errors**.

During build, `scripts/prefetch_pdf_models.py` downloads unstructured /
Hugging Face layout models (e.g. YOLOX ONNX). Those libraries use **certifi**,
which does **not** automatically include OS/MITM roots. The Dockerfile therefore
runs `scripts/inject_system_cas_into_certifi.py` so certifi’s bundle includes
`/etc/ssl/certs/ca-certificates.crt` (with your inspection CA).

If prefetch still fails with `SSL: CERTIFICATE_VERIFY_FAILED` against
`huggingface.co`, rebuild with:

```bash
docker build -t sauron --build-arg SAURON_PREFETCH_INSECURE_SSL=1 .
```

That disables TLS verify **only** for the prefetch step (models are still
baked into the image; runtime stays on normal verify + your CA).

`Trusted_Root_CAs.pem` is gitignored so environment-specific CAs are not
committed by accident. Force-add it if you intentionally want it in the repo.
