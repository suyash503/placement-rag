export type Role = "user" | "assistant";

export interface Citation {
  index: number;
  doc_type: "record" | "company";
  company: string;
  college?: string | string[] | null;
  year?: number | number[] | null;
  role?: string | null;
  package_lpa?: number | null;
  cgpa_cutoff?: number | null;
  branches: string[];
  selection_rounds: string[];
  active_backlogs?: string | null;
  job_location?: string | null;
  text: string;
}

export interface ScoredDoc {
  doc: string;
  score: number;
}

export interface RetrievalTrace {
  mode?: string;
  filters?: Record<string, unknown>;
  filter_notes?: string[];
  filter_relaxed?: boolean;
  condensed_question?: string;
  candidate_k?: number;
  final_k?: number;
  candidates_found?: number;
  returned?: string[];
  rerank_top?: ScoredDoc[];
  vector_only?: ScoredDoc[];
  fulltext_only?: ScoredDoc[];
  timings_ms?: Record<string, number>;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  citations?: Citation[];
  trace?: RetrievalTrace;
  latencyMs?: number;
  streaming?: boolean;
  error?: string;
}

export interface Stats {
  documents: number;
  records: number;
  company_profiles: number;
  companies: number;
  colleges: number;
  years: number[];
  package_min?: number | null;
  package_max?: number | null;
  indexes: { name: string; type: string; status?: string }[];
}
