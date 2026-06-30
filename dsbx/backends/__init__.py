"""Backend implementations (filled in from Wave 1 onward).

Planned backends:
- hf          : HuggingFace transformers, full-vocab logits (white box, on dsbx-host)
- llamacpp    : llama-server over HTTP, top-k logprobs (fast, on dsbx-host)
- openai_compat: Fireworks / NVIDIA NIM / OpenRouter / LM Studio (cloud top-k)
"""
