/**
 * Wire shapes returned by the dsbx web middleware.
 *
 * These mirror the Pydantic models in ``decoding_sandbox/web/schemas.py``
 * and ``decoding_sandbox/server/schemas.py``. Keeping them in TypeScript
 * (rather than codegen) is intentional -- the API surface is small and the
 * types stay close to the components that consume them.
 */

export interface Capabilities {
  name: string;
  full_vocab: boolean;
  prompt_logprobs: boolean;
  max_top_logprobs: number;
  can_force_token: boolean;
  notes: string;
  eos_token_ids: number[];
  /**
   * Begin-of-sequence marker token id(s) the model uses -- either a
   * true BOS (Llama family ``<|begin_of_text|>``) or a document-
   * boundary token reused as such (Qwen Base uses ``<|endoftext|>``
   * for both ends). Empty array on models with no canonical BOS or
   * on cloud backends that tokenize server-side (and therefore can't
   * be helped by client-side prepend). The Decode workbench's "fill
   * BOS" helper greys out when this is empty and drops the listed
   * ids into the prepend chip-input when clicked.
   */
  bos_token_ids: number[];
  // Provider-specific /v1/completions extension flags. Surfaced so the
  // UI can adapt without hard-coding provider names: ``supports_ignore_eos``
  // unlocks the "respect EOS" checkbox for Fireworks; ``supports_perf_metrics``
  // shows the server-timings panel; ``supports_sampling_mask`` adds the
  // "eligible after filters" column to the steps table;
  // ``supports_raw_output`` surfaces the "what the model saw" panel;
  // ``supports_service_tier`` exposes the priority/default selector.
  supports_ignore_eos: boolean;
  supports_perf_metrics: boolean;
  supports_service_tier: boolean;
  supports_sampling_mask: boolean;
  supports_raw_output: boolean;
  supports_logit_bias: boolean;
  /**
   * When true, the backend accepts a non-empty
   * ``prepend_token_ids`` on score_prompt / generate / inspect --
   * those ids get spliced in FRONT of the tokenized prompt before
   * scoring, which unlocks the pedagogical "predict position 0 from
   * BOS" workflow (otherwise unscorable: autoregressive models have
   * no prior to condition on at position 0). True for local backends
   * (HF / llamacpp-py / dsbx-host-py) AND for cloud backends with a
   * configured-and-fetched HF tokenizer (Fireworks gpt-oss, Qwen,
   * etc. -- see ``supports_local_tokenize``). The Decode workbench's
   * prepend chip-input gates on this flag and the "fill BOS" button
   * is disabled when it's false.
   */
  supports_prepend_token_ids: boolean;
  /**
   * When true, the web layer's ``/api/v1/tokenize`` endpoint returns
   * a real per-text token id list for this backend. Always true for
   * local backends; true for cloud backends once their per-model HF
   * tokenizer.json has been fetched (lazy, on first use, cached on
   * disk via the standard HuggingFace cache). Drives the live token
   * preview under the prompt textarea in the Decode workbench: when
   * false the preview is hidden (would otherwise show one synthetic
   * id for the whole prompt, which teaches nothing); when true the
   * UI debounces the user's typing and renders one chip per token.
   */
  supports_local_tokenize: boolean;
  /**
   * When true, ``include_prompt`` runs as a single ``echo=true`` +
   * ``stream=true`` request instead of two separate calls
   * (``score_prompt`` + per-token stream). Halves provider RPS for
   * the include-prompt path. The UI uses this flag to expose the
   * ``echo_last`` knob (only meaningful when the combined path is
   * actually used).
   */
  supports_combined_echo_stream: boolean;
  /**
   * When true, the backend is REGISTERED but inert: generate-stream
   * requests against it are rejected with a 400 by the middleware,
   * and the backend picker renders it as a disabled option with
   * ``notes`` as the tooltip explanation. Set on chat-only OpenAI-
   * compat providers (NIM / OpenRouter) until proper chat-mode UI
   * lands. Default ``false`` for every other backend.
   */
  generation_disabled: boolean;
}

export interface BackendInfo {
  name: string;
  label: string;
  family: 'remote' | 'cloud' | 'local';
  capabilities: Capabilities | null;
  available: boolean;
  note: string;
  // Model the backend is currently running (or its default for deferred-load
  // backends). Null when unknown (e.g. a remote we haven't pinged yet).
  loaded_model: string | null;
  // Picker options for the model dropdown. Cloud providers carry their
  // curated catalogue here; remote / local backends carry a single-element
  // list with ``loaded_model``.
  suggested_models: string[];
  // Whether the UI lets the user pick a different model per request. Only
  // cloud providers set this true today; flipping it for a remote backend
  // would require restarting ``dsbx serve`` on dsbx-host.
  model_editable: boolean;
}

export interface InfoResponse {
  engine_version: string;
  server_label: string;
  default_backend: string;
  backends: BackendInfo[];
}

export interface ModelsResponse {
  backend: string;
  models: string[];
  source: 'live' | 'cached' | 'static' | 'fallback';
  fetched_at: number | null;
  cache_ttl_s: number;
  note: string;
}

export interface TokenCandidate {
  token_id: number;
  text: string;
  logprob: number | null;
  rank: number;
  is_special: boolean;
  /**
   * Number of tokens that survived the server-side sampling filter
   * stack (Fireworks `sampling_mask=count`). Only meaningful for the
   * full distribution at a position -- the value is replicated onto
   * every candidate at that position by the backend so the UI can
   * read `candidates[0].sampling_mask_count` indiscriminately.
   * `null` on backends that don't report it.
   */
  sampling_mask_count?: number | null;
}

export interface Watched {
  token_id: number;
  candidate: TokenCandidate;
}

export interface StepResult {
  position: number;
  candidates: TokenCandidate[];
  is_full_vocab: boolean;
  chosen: TokenCandidate | null;
  context_text: string | null;
  watched: Watched[];
}

// ``ResolvedWatch`` / ``InspectResponse`` used to live here. They are
// gone -- the inspect endpoint was deleted (plan: Unify Decode
// Workbench Phase 3) and the Decode workbench page reconstructs watch
// column labels locally from what it sent (``watchTexts`` /
// ``watchIds`` / ``watchEos``), so the round-trip ``ResolvedWatch``
// payload was no longer needed.

export interface SamplerSpec {
  name: string;
  params: Record<string, number | null>;
}

export interface KeptEntry {
  token_id: number;
  prob: number;
}

export interface SamplerDecision {
  token_id: number;
  token_text: string;
  kept: KeptEntry[];
  greedy_token_id: number | null;
  note: string;
}

export interface GenStep {
  step: number;
  tokens_before: number[];
  step_result: StepResult;
  decision: SamplerDecision;
  stop_reason: string | null;
}

export interface PromptScorePayload {
  steps: StepResult[];
  is_full_vocab: boolean;
  prompt_logprobs: boolean;
  note: string;
}

export interface UsagePayload {
  // HTTP calls against an upstream provider this run consumed.
  // Includes retries on purpose: a 429-then-200 retry counts as 2,
  // matching what the provider's rate-limit bucket saw.
  requests: number;
  // Provider-reported counts (preferred) or local fallback for
  // backends without an upstream tokenizer. ``null`` means
  // "unknown" -- e.g. chat-streaming providers that don't report
  // a usage block.
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  // Short human-readable advisories the backend wanted the user to
  // see, e.g. "respect_eos=False is unsupported by this cloud
  // provider". Rendered next to the counters.
  notes: string[];
}

/**
 * Server-side performance metrics, surfaced via the ``perf`` SSE frame.
 *
 * Populated only for backends that advertise ``supports_perf_metrics``
 * (Fireworks today). The shape is opaque -- keys come straight from the
 * upstream's ``perf_metrics`` block in the response body / final stream
 * chunk. The UI renders known keys with friendly labels and falls back
 * to ``key: value`` for anything unknown.
 *
 * Typical keys (Fireworks):
 * - ``prompt-tokens`` / ``cached-prompt-tokens``
 * - ``server-time-to-first-token`` (seconds)
 * - ``server-processing-time`` (seconds)
 * - ``prefill-duration`` / ``generation-duration``
 * - ``speculation-acceptance`` (per-position acceptance, dedicated only)
 * - ``backend-host`` / ``deployment`` (dedicated deployments only)
 */
export interface PerfMetricsPayload {
  // Free-form: keys depend on the provider and deployment kind. Stored
  // as ``unknown`` so consumers must check the type at render time.
  [key: string]: unknown;
}

/**
 * Provider-side "what the model actually saw" diagnostics.
 *
 * Mirrors the Fireworks `raw_output` block 1:1; we type the well-known
 * keys (`prompt_fragments`, `prompt_token_ids`, `grammar`) so the UI
 * can render them with structure, and keep the rest as `unknown` so a
 * provider adding a new key doesn't force a wire-schema bump.
 */
export interface RawOutputPayload {
  prompt_fragments?: string[] | null;
  prompt_token_ids?: number[] | null;
  grammar?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export type SSEEvent =
  | { event: 'step'; step: GenStep }
  | {
      event: 'prompt_score';
      steps: StepResult[];
      is_full_vocab: boolean;
      prompt_logprobs: boolean;
      note: string;
    }
  | { event: 'perf'; metrics: PerfMetricsPayload }
  | { event: 'raw_output'; payload: RawOutputPayload }
  | ({ event: 'usage' } & UsagePayload)
  | { event: 'done'; stop_reason: string | null; error?: string | null }
  | { event: 'round'; round: SpecRound }
  | {
      event: 'done';
      total_proposed?: number;
      total_accepted?: number;
      total_emitted?: number;
      rounds?: number;
      completion?: string;
      error?: string | null;
    };

export interface SpecRound {
  step: number;
  proposed: TokenCandidate[];
  accepted: number;
  correction: TokenCandidate | null;
  emitted_ids: number[];
}

export interface ManualSnapshot {
  session_id: string;
  backend: string;
  prompt: string;
  prompt_ids: number[];
  generated_ids: number[];
  generated_text: string;
  top_k: number;
  distribution: StepResult;
  can_force_token: boolean;
  // Per-emitted-token linear probability (or null for forced/load-loaded
  // tokens). Indexed in lockstep with ``generated_ids``. The UI uses this
  // to color the running completion text.
  generated_probs: (number | null)[];
  // Printable per-token text. Lockstep with ``generated_ids``. The UI uses
  // this so it can render each token as its own ``<span>`` without an
  // extra ``/piece`` round trip per token.
  generated_pieces: string[];
  model: string | null;
}

export interface ManualTranscript {
  prompt: string;
  backend: string;
  prompt_ids: number[];
  generated_ids: number[];
  generated_text: string;
  top_k: number;
  model: string | null;
}

export interface ProbeRow {
  provider: string;
  model: string;
  chat_logprobs: string;
  prompt_logprobs: string;
}

export interface ProbeResponse {
  rows: ProbeRow[];
  fresh: boolean;
  cached_at: number | null;
}

/**
 * Upstream-request log shapes -- mirror ``decoding_sandbox.web.logs_api``.
 *
 * Every outgoing HTTP call the middleware makes to a real backend (dsbx-host dsbx
 * server, Fireworks/NIM/OpenRouter, local llama-server) lands as one row.
 * Streaming responses are merged into a single ``response_body`` plus a list
 * of the raw frames in ``stream_chunks``.
 */
export interface LogSummary {
  id: string;
  // ISO-8601 timestamp string; the UI parses it with Date(...) when needed.
  timestamp: string;
  backend_name: string;
  backend_family: string;
  provider_name: string | null;
  method: string;
  upstream_path: string;
  response_status_code: number | null;
  is_streaming: boolean;
  latency_ms: number | null;
  ttft_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  model_resolved: string | null;
  // List-page snippet (truncated server-side to ~240 chars). Use ``getLog``
  // to fetch the full untruncated text.
  completion_text: string | null;
  stop_reason: string | null;
  error_message: string | null;
}

export interface LogDetail extends LogSummary {
  upstream_url: string;
  request_headers: Record<string, string> | null;
  request_body: unknown;
  request_body_text: string | null;
  response_headers: Record<string, string> | null;
  response_body: unknown;
  response_body_text: string | null;
  stream_chunks: unknown[] | null;
}

export interface LogListResponse {
  items: LogSummary[];
  next_cursor: string | null;
  has_more: boolean;
}

export interface LogStats {
  total: number;
  streaming: number;
  non_streaming: number;
  error_count: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  avg_latency_ms: number | null;
  avg_ttft_ms: number | null;
}

export interface LogListParams {
  cursor?: string | null;
  limit?: number;
  backend?: string | null;
  provider?: string | null;
  status_code?: number | null;
  is_error?: boolean | null;
  since?: string | null;
}
