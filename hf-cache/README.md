# Optional: copy a pre-populated Hugging Face cache here for air-gapped builds.
# On a machine with HF access:
#   huggingface-cli download nomic-ai/nomic-embed-text-v1
#   cp -a ~/.cache/huggingface/. hf-cache/
# Then: docker compose build
