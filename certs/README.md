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
