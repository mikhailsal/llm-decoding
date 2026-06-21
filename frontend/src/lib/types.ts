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
}

export interface InfoResponse {
  engine_version: string;
  server_label: string;
  default_backend: string;
  backends: BackendInfo[];
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

export type SSEEvent =
  | { event: 'step'; step: GenStep }
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
}

export interface ManualTranscript {
  prompt: string;
  backend: string;
  prompt_ids: number[];
  generated_ids: number[];
  generated_text: string;
  top_k: number;
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
