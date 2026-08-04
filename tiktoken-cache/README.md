# Optional pre-seeded tiktoken cache

The Docker build normally downloads and verifies every encoding supported by
the installed `tiktoken` package. For a build that itself has no network
access, copy a populated `TIKTOKEN_CACHE_DIR` into this directory before
building. The directory contents are copied to `/app/.cache/tiktoken`.

Do not remove the build-time offline verification: it prevents an image from
being published when LightRAG would attempt to download tokenizer vocabulary
data at runtime.
