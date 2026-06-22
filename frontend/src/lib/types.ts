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

export interface ResolvedWatch {
  label: string;
  token_id: number;
  source: 'text' | 'id' | 'eos';
  piece: string;
}

export interface InspectResponse {
  steps: StepResult[];
  watches: ResolvedWatch[];
  is_full_vocab: boolean;
  prompt_logprobs: boolean;
  note: string;
}

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

export type SSEEvent =
  | { event: 'step'; step: GenStep }
  | {
      event: 'prompt_score';
      steps: StepResult[];
      is_full_vocab: boolean;
      prompt_logprobs: boolean;
      note: string;
    }
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
