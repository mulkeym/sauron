"""Download the unstructured hi_res layout + table-transformer models at BUILD
time so the runtime image needs no network for scanned-PDF OCR. Run during
docker build, BEFORE HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set."""
import sys


def main() -> int:
    # partition_pdf with hi_res pulls the layout + table models on first use;
    # invoking it once here caches them into the image layer.
    from unstructured.partition.pdf import partition_pdf
    try:
        partition_pdf(
            filename="tests/fixtures/pdf/tiny_smoke.pdf",
            strategy="hi_res",
            infer_table_structure=True,
        )
    except Exception as e:
        print(f"prefetch failed: {e}", file=sys.stderr)
        return 1
    print("pdf models prefetched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
