"""Core data model shared by every backend and UI.

These types are deliberately backend-agnostic: a full-vocab HF forward pass, a
top-k llama.cpp response, and a cloud provider's top_logprobs all reduce to the
same ``StepResult`` so the inspect/generate/manual UIs never special-case a
backend -- they only read ``Capabilities`` to decide what to show.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenCandidate:
    """One candidate token in a distribution at a single position.

    ``is_special`` is set by backends that can tell (HF via
    ``tokenizer.all_special_ids``, llama-cpp-py via ``Llama.token_eos()`` +
    the ``<|...|>`` heuristic). The renderer uses it to colour the token
    distinctively so the user can immediately see EOS/BOS/PAD without
    eyeballing strings like ``<|endoftext|>``. Backends that don't expose
    this info leave it ``False`` -- the renderer falls back to a pattern
    check on the text.

    ``sampling_mask_count`` is the number of vocabulary tokens that
    survived the server's sampler filters (top_k / top_p / min_p /
    typical / mirostat) at this position. It comes from the Fireworks
    NewLogProbs response's ``sampling_mask: 'count'`` field; backends
    that don't support that flag leave it ``None``. The generate /
    inspect table renders this as a separate "eligible after filters"
    column so the user can see when a tight top_p genuinely cut the
    candidate set down to a handful vs when the filter was effectively
    a no-op (count ≈ vocab size).
    """

    token_id: int
    text: str
    logprob: float
    rank: int  # 0 = most likely
    is_special: bool = False
    sampling_mask_count: int | None = None

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)


@dataclass
class StepResult:
    """The model's predicted distribution at one position.

    Used both for *inspection* (``chosen`` = the actual next token already in the
    text, so we can show the probability the model assigned to reality) and for
    *generation* (``chosen`` = the token the sampler picked).
    """

    position: int
    candidates: list[TokenCandidate]  # ranked, most likely first
    is_full_vocab: bool
    chosen: TokenCandidate | None = None
    # The token text at this position (the context token being conditioned on),
    # handy for rendering inspect rows. Optional.
    context_text: str | None = None
    # Probability of specific "watch" tokens at this position, even if they fall
    # outside the top-k. A candidate with rank == -1 / nan logprob means unknown
    # (token outside a non-full-vocab backend's returned top-k).
    watched: dict[int, TokenCandidate] = field(default_factory=dict)

    @property
    def top(self) -> TokenCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def confidence(self) -> float:
        """Max probability (top-1) -- the model's confidence at this position."""
        t = self.top
        return t.prob if t else 0.0

    def find(self, token_id: int) -> TokenCandidate | None:
        for c in self.candidates:
            if c.token_id == token_id:
                return c
        return None


@dataclass
class Capabilities:
    """What a backend can do, so the UI can adapt instead of guessing.

    ``eos_token_ids`` lists every token id the backend believes terminates a
    generation. Set non-empty by backends that expose it (HF reads
    ``model.config.eos_token_id``, llama-cpp-py reads ``Llama.token_eos()``).
    The ``generate`` engine treats any chosen token id in this set as an
    implicit stop, so a base model that wants to emit ``<|endoftext|>``
    actually halts instead of running until ``--max-tokens``.

    The ``supports_*`` flags below are mirrors of provider-specific
    /v1/completions extensions (currently Fireworks-only). They let the
    UI adapt without hard-coding "if backend.name == 'fireworks'" checks
    everywhere: ``supports_ignore_eos`` unlocks the ``respect EOS``
    checkbox; ``supports_perf_metrics`` shows the server-timings panel;
    ``supports_sampling_mask`` enables the "eligible after filters"
    column in the generation steps table; ``supports_raw_output``
    surfaces the "what the model actually saw" panel with rendered
    prompt fragments + grammar; ``supports_service_tier`` exposes the
    priority/default selector.
    """

    name: str
    full_vocab: bool  # exact distribution over the entire vocabulary
    prompt_logprobs: bool  # can score every prompt token (whole context)
    max_top_logprobs: int  # how many candidates per position it can return
    can_force_token: bool = False  # supports manual/forced token decoding
    notes: str = ""
    eos_token_ids: tuple[int, ...] = ()
    # Token ids the model considers a "beginning of sequence" marker --
    # either a true BOS (Llama family ``<|begin_of_text|>``) or a
    # document-boundary token reused as such (Qwen Base uses
    # ``<|endoftext|>`` for both ends). Empty tuple means "this model
    # has no canonical BOS we can recommend"; the UI's "fill BOS"
    # helper greys out in that case. Used by the pedagogical
    # ``prepend_token_ids`` workflow: when the user clicks "fill BOS"
    # we drop these ids into the prepend chip-input so they can see
    # the BOS-conditioned distribution for what would otherwise be an
    # unscorable position 0. Backends discover the value differently
    # (HF: tokenizer.bos_token_id; llama-cpp-py: Llama.token_bos();
    # openai-compat: a small known-model table; remote: forwarded from
    # the upstream dsbx serve's /v1/info). When discovery returns
    # ``None`` we leave the tuple empty.
    bos_token_ids: tuple[int, ...] = ()
    # When true, ``Backend.score_prompt`` and ``Backend.stream_native``
    # accept a non-empty ``prepend_token_ids`` argument: those tokens
    # are concatenated BEFORE the tokenized prompt, so the model sees
    # ``prepend_token_ids + tokenize(prompt)`` as one continuous
    # sequence. Used by the "predict position 0 from BOS" workflow.
    # False for cloud providers that take a plain ``prompt: str`` and
    # tokenize server-side AND we don't have a local replica of the
    # model's tokenizer. With a real local HF tokenizer (see
    # ``supports_local_tokenize``) cloud backends flip this to True
    # too -- we switch to token-array prompt mode (``"prompt": [int,
    # ...]``) when the call carries a non-empty ``prepend_token_ids``,
    # which Fireworks et al. support out of the box. The frontend
    # gates the prepend chip-input on this flag.
    supports_prepend_token_ids: bool = False
    # When true, the backend can return a real per-text token id list
    # (``backend.tokenize(text) -> list[int]`` is NOT a single-intern
    # stub). Local backends (HF / llamacpp-py / dsbx-host-py) are always
    # true; cloud backends flip true once their per-model HF tokenizer
    # has been fetched via ``hf_hub_download``. The Decode workbench
    # uses this flag to decide whether to render the live token
    # preview under the prompt textarea: a real token list makes it
    # educational ("see how your text becomes tokens"), the synthetic
    # stub does not.
    supports_local_tokenize: bool = False
    # Provider-specific completion extensions (Fireworks today). Stay
    # False for HF / llamacpp / chat-only cloud providers; surfaced over
    # the wire so the browser doesn't need to know which provider it is
    # talking to to decide which UI affordances to enable.
    supports_ignore_eos: bool = False
    supports_perf_metrics: bool = False
    supports_service_tier: bool = False
    supports_sampling_mask: bool = False
    supports_raw_output: bool = False
    # Per-request ``logit_bias`` map (token_id -> bias in [-100, 100]).
    # OpenAI Completions has always supported it; we still gate the UI
    # editor on this so providers that ignore the field don't show a
    # knob that does nothing.
    supports_logit_bias: bool = False
    # ``include_prompt`` mode in a single round trip when true (Phase 5).
    # When false the web layer falls back to two requests
    # (``score_prompt`` + ``stream_native``). Surfaced so the UI can
    # show an "echo_last" knob only where the combined path runs.
    supports_combined_echo_stream: bool = False
    # When true, the backend is REGISTERED but refuses to generate. The
    # current trigger is "chat-only OpenAI-compat provider" (NIM /
    # OpenRouter -- ``ProviderConfig.has_completions = false``): the
    # per-step "growing user message" emulation we used to do here
    # produced N independent first-responses instead of a real
    # continuation, so it's gated off until a proper chat-mode UI lands
    # (separate PR). The web route ``/api/v1/generate/stream`` enforces
    # the gate by returning 400 when this flag is true; the frontend
    # backend picker renders such entries as disabled options and uses
    # the ``notes`` field as the tooltip explanation. Default ``False``
    # for every other backend.
    generation_disabled: bool = False


@dataclass
class RawOutputInfo:
    """Provider diagnostics returned with ``raw_output: true`` requests.

    Fireworks' ``raw_output`` block answers the question "what did the
    model actually see / produce?" -- *before* any post-processing,
    template rendering or grammar-constrained decoding hides it. The
    sandbox stores the bits we know how to render in dedicated fields
    and keeps the full payload under ``raw`` so the UI's "what the
    model saw" panel can pretty-print every key the provider chose to
    include, even ones we haven't typed yet.

    Concrete fields we surface today:

    * ``prompt_fragments`` -- the prompt string broken into the
      fragments the templating engine fed to the model. Useful for
      sanity-checking custom chat templates: if your role tags
      disappeared, this is where you see it.
    * ``prompt_token_ids`` -- the actual tokenized prompt the model
      saw, *including* injected BOS / system tokens. The
      ``--watch-id`` workflow can compare these against the user's
      typed prompt to detect silent tokenizer mismatches.
    * ``grammar`` -- the grammar object the server compiled (if any).
      With response_format / json mode this tells you exactly which
      constrained-decoding FSM was active.

    Everything else from the provider lands in ``raw`` verbatim.
    """

    prompt_fragments: list[str] | None = None
    prompt_token_ids: list[int] | None = None
    grammar: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


__all__ = ["Capabilities", "RawOutputInfo", "StepResult", "TokenCandidate", "field"]
