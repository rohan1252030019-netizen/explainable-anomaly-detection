/**
 * frontend/src/lib/api.ts
 * Type-safe API client connecting the Next.js frontend to the FastAPI backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8001";
// Matches XAUD_API_KEY on the server.  Empty means the API is open (the local
// default), so nothing is sent and existing development setups are unchanged.
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

export interface ColumnInfo {
  name: string;
  dtype: string;
  non_null_count: number;
  null_count: number;
  null_pct: number;
  unique_count: number;
  is_numeric: boolean;
  sample_values?: unknown[];
}

export interface DatasetSummary {
  loaded: boolean;
  dataset_name?: string;
  n_rows?: number;
  n_cols?: number;
  numerical_cols?: string[];
  categorical_cols?: string[];
  total_missing?: number;
  duplicate_rows?: number;
  memory_mb?: number;
  columns?: ColumnInfo[];
  preview?: Record<string, unknown>[];
  feature_matrix_ready?: boolean;
  n_preprocessed_features?: number;
  label_col?: string | null;
  warnings?: string[];
}

export interface ScoreHistBin {
  bin_start: number;
  bin_end: number;
  normal_count: number;
  anomaly_count: number;
}

export interface ScoreTimelinePoint {
  index: number;
  score: number;
  is_anomaly: boolean;
  severity: string;
}

export interface SeverityBandLegend {
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  count: number;
  share: number;
  rank_range: [number, number];
  definition: string;
}

export interface ScoreNormalization {
  method: "quantile" | "rank" | "minmax";
  offset: number;
  scale: number;
  low_quantile: number;
  high_quantile: number;
}

export interface DetectionOverview {
  ready: boolean;
  model_name?: string;
  n_total?: number;
  n_anomalies?: number;
  anomaly_rate?: number;
  avg_score?: number;
  median_anomaly_score?: number;
  max_score?: number;
  threshold_val?: number;
  threshold_method?: string;
  severity_counts?: {
    CRITICAL: number;
    HIGH: number;
    MEDIUM: number;
    LOW: number;
  };
  /** What each severity tier actually means in this run. */
  severity_bands?: {
    mode: string;
    levels: SeverityBandLegend[];
    note: string;
  } | null;
  /** The score mapping used, so 0.8 can be interpreted. */
  score_normalization?: ScoreNormalization | null;
  score_histogram?: ScoreHistBin[];
  score_timeline?: ScoreTimelinePoint[];
  message?: string;
}

export interface AnomalyRecord {
  rank: number | null;
  /** Position in the ORIGINAL dataset. Required to open an investigation. */
  original_index?: number;
  normalized_score: number;
  anomaly_score: number;
  prediction: number;
  is_anomaly: boolean;
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NORMAL";
  model: string;
  [key: string]: unknown;
}

export interface AnomalyListResponse {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  items: AnomalyRecord[];
}

export interface FeatureComparisonItem {
  feature: string;
  observed: unknown;
  typical_median: number;
  typical_iqr: number;
  typical_min?: number;
  typical_max?: number;
  deviation: string;
  direction: string;
  contribution: string;
  interpretation: string;
}

/** A concrete "what would have to change" suggestion for one anomaly. */
export interface CounterfactualChange {
  feature: string;
  direction: "increase" | "decrease";
  /** Observed value in the dataset's ORIGINAL units. */
  observed_value?: number;
  suggested_value?: number;
  change_pct?: number | null;
  scaled_from: number;
  scaled_to: number;
  magnitude_scaled: number;
  importance_weight: number;
}

export interface Counterfactual {
  found: boolean;
  reason?: string | null;
  neighbour_index?: number | null;
  distance?: number | null;
  score_before?: number | null;
  score_after?: number | null;
  score_reduction?: number | null;
  score_threshold?: number | null;
  quality?: {
    label: "STRONG" | "PARTIAL" | "WEAK" | "UNVERIFIED" | "NONE";
    detail: string;
  };
  n_changes?: number;
  changes?: CounterfactualChange[];
  intervention_cost?: number;
  immutable_features_enforced?: string[];
  text?: string;
  caveat?: string;
  row_id?: number;
}

/** Behavioural check that the explanation's ranking actually drives the score. */
export interface ExplanationVerification {
  overall:
    | "VERIFIED"
    | "BEHAVIOURALLY FAITHFUL"
    | "PARTIALLY VERIFIED"
    | "UNVERIFIED"
    | "UNKNOWN";
  checks_run: number;
  summary: string;
  faithfulness?: {
    deletion_auc_explained: number | null;
    deletion_auc_random: number | null;
    deletion_auc_random_std?: number;
    improvement?: number;
    relative_improvement: number | null;
    n_features: number;
    curve?: number[];
    verdict: string;
  };
  additivity?: {
    checked: boolean;
    passed: boolean;
    max_abs_error?: number | null;
    note?: string;
  };
}

export interface InvestigationDetail {
  row_id: number;
  rank: number | null;
  model_name: string;
  normalized_score: number;
  is_anomaly: boolean;
  severity: string;
  threshold: number;
  explanation_text: string;
  confidence_note: string;
  top_features: Array<{
    feature: string;
    combined_score: number;
    deviation_direction: string;
    robust_z_score: number;
    feature_type?: string;
  }>;
  feature_comparison: FeatureComparisonItem[];
  counterfactual?: Counterfactual | null;
  verification?: ExplanationVerification | null;
  raw_record: Record<string, unknown>;
}

export interface HoldoutMetrics {
  available: boolean;
  reason?: string | null;
  model?: string;
  summary?: string;
  note?: string;
  n_fit?: number;
  n_test?: number;
  test_size?: number;
  random_state?: number;
  normalization?: string;
  score_mean_in_sample?: number;
  score_mean_out_of_sample?: number;
  flag_rate_in_sample?: number;
  flag_rate_out_of_sample?: number;
  flag_rate_drift?: number;
  in_sample?: MetricBlock;
  out_of_sample?: MetricBlock;
  optimism_gap?: Record<string, number | null>;
}

export interface MetricBlock {
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  roc_auc?: number | null;
  pr_auc?: number | null;
  n_true_anomalies?: number;
  n_detected_anomalies?: number;
  error?: string;
}

export interface HoldoutComparison {
  test_size: number;
  random_state: number;
  note: string;
  table: Array<Record<string, number | string | null>>;
}

export interface StabilityReport {
  n_fits: number;
  mean_rank_correlation?: number | null;
  min_rank_correlation?: number | null;
  top_k_retention?: number | null;
  per_fit_leader?: string[];
  verdict?: string;
  seeds?: number[];
  model?: string;
  n_anomalies_probed?: number;
  top_k_features?: Array<{
    feature: string;
    appeared_in_top_k: number;
    share: number;
  }>;
}

export interface GlobalImportanceResponse {
  ready: boolean;
  n_explained?: number;
  top_driver?: string;
  global_importance?: Array<{
    feature: string;
    mean_combined_score: number;
    frequency: number;
    mean_z_score: number;
  }>;
  message?: string;
}

export interface PcaAnalyticsResponse {
  variance_explained: number[];
  points: Array<{
    id: number;
    pc1: number;
    pc2: number;
    is_anomaly: boolean;
    score: number;
    severity: string;
  }>;
}

export interface ModelBenchmarkResponse {
  comparison_table: Array<{
    Model: string;
    Anomalies?: number | string;
    /** Backend serialises benchmark metrics as formatted strings. */
    "Score Sep."?: number | string;
    "Fit Time (s)"?: number | string;
    "Runtime (s)"?: number | string;
    F1?: number | string;
    Precision?: number | string;
    Recall?: number | string;
    "ROC-AUC"?: number | string;
    "PR-AUC"?: number | string;
    Status?: string;
    [key: string]: unknown;
  }>;
  all_model_consensus: number;
  disagreed_count: number;
  agreed_normal: number;
  model_anomaly_counts: Array<{ model: string; anomaly_count: number }>;
  pairwise_jaccard: Array<{
    model_a: string;
    model_b: string;
    jaccard: number;
  }>;
}

export interface ExperimentRecord {
  id: number;
  model: string;
  threshold_method: string;
  n_anomalies: number;
  anomaly_rate?: number;
  fit_time_sec?: number;
  timestamp: string;
}

export interface PreprocessingSummary {
  scaler: string;
  n_features: number;
  label_col?: string | null;
  [key: string]: unknown;
}

export interface GlobalImportanceEntry {
  feature: string;
  mean_combined_score: number;
  frequency: number;
  mean_z_score: number;
}

export interface AuditReport {
  title: string;
  framework: string;
  generated_at: string;
  dataset: {
    name: string;
    rows?: number;
    features_preprocessed: number;
    scaler: string;
  };
  detection: {
    model: string;
    threshold_method: string;
    threshold_value?: number;
    total_observations: number;
    anomalies_detected: number;
    anomaly_rate?: number;
    severity_distribution?: Partial<
      Record<"CRITICAL" | "HIGH" | "MEDIUM" | "LOW", number>
    >;
  };
  top_drivers?: Array<{
    feature: string;
    mean_combined_score?: number;
    frequency?: number;
  }>;
  methodology_note: string;
}


export interface DatasetProfileResponse {
  health: {
    score: number;
    interpretation: string;
    deductions: string[];
  };
  schema: {
    numerical_columns: string[];
    categorical_columns: string[];
    binary_columns: string[];
    datetime_columns: string[];
    id_candidates: string[];
    constant_columns: string[];
  };
  quality: {
    total_cells: number;
    missing_cells: number;
    missing_pct: number;
    duplicate_rows: number;
    duplicate_pct: number;
    n_constant_features: number;
    warnings: string[];
  };
  collinearity: {
    condition_number: number;
    high_correlations: Array<{
      feature_a: string;
      feature_b: string;
      pearson_r: number;
    }>;
    severely_collinear: boolean;
  };
  recommendations: string[];
}

export interface ModelCapability {
  family?: string;
  time_complexity?: string;
  space_complexity?: string;
  scaling_required?: boolean;
  sensitive_to_dim?: boolean;
  explanation_methods?: string[];
  score_direction?: string;
  strengths?: string;
  limitations?: string;
}

export interface EnsembleRunRequest {
  models: string[];
  aggregation_method?: "rank_borda" | "weighted_score";
  weights?: Record<string, number>;
  min_consensus_votes?: number;
}

export interface EnsembleResponse {
  success: boolean;
  ensemble: {
    ensemble_scores: number[];
    consensus_predictions: number[];
    consensus_counts: number[];
    consensus_strength: number[];
    disagreement: {
      pairwise_jaccard_distance: Record<string, Record<string, number>>;
      spearman_rank_correlation: Record<string, Record<string, number>>;
      mean_pairwise_disagreement: number;
    };
    method: string;
    models_used: string[];
  };
  overview: DetectionOverview;
}

export interface FeatureDriftResult {
  feature: string;
  type: "numerical" | "categorical";
  /** Population Stability Index (numerical features). */
  psi?: number;
  ks_statistic?: number;
  p_value?: number;
  statistically_significant?: boolean;
  /** Jensen-Shannon distance (categorical features). */
  js_distance?: number;
  unique_cats_ref?: number;
  unique_cats_cur?: number;
  ref_mean?: number;
  cur_mean?: number;
  /** The backend's per-feature verdict; overall uses HIGH/MODERATE/NONE. */
  drift_status: "HIGH" | "MODERATE" | "STABLE";
}

export interface DriftReport {
  ready?: boolean;
  message?: string;
  overall_severity?: "NONE" | "LOW" | "MODERATE" | "HIGH";
  mean_feature_psi?: number;
  n_features_evaluated?: number;
  features_high_drift?: string[];
  features_moderate_drift?: string[];
  feature_reports?: FeatureDriftResult[];
  score_drift?: {
    psi?: number;
    drift_status?: string;
    ref_mean?: number;
    cur_mean?: number;
    [key: string]: unknown;
  } | null;
  missingness_drift?: Array<Record<string, unknown>>;
  /** Remediation guidance (the backend calls this `actions`). */
  actions?: string[];
}

export interface SyntheticBenchmarkResultItem {
  model: string;
  roc_auc: number;
  pr_auc: number;
  precision_at_k: number;
  rank: number;
}

export interface SyntheticBenchmarkResponse {
  ready?: boolean;
  message?: string;
  experiment_type?: string;
  anomaly_type?: "point" | "multivariate" | "cluster" | "contextual";
  contamination?: number;
  difficulty?: "easy" | "medium" | "hard";
  n_injected?: number;
  results_table?: SyntheticBenchmarkResultItem[];
  best_model?: string;
  note?: string;
}

export interface PersistentExperimentRecord {
  experiment_id: string;
  timestamp: string;
  dataset_name: string;
  /** Full SHA-256 of the dataset the run used. */
  dataset_fingerprint: string;
  /** 16-character prefix retained for older clients. */
  dataset_hash?: string;
  model_name: string;
  parameters: Record<string, unknown>;
  random_state?: number;
  threshold_method?: string;
  threshold_value?: number;
  fit_time_sec?: number;
  n_anomalies_flagged?: number;
  anomaly_rate?: number;
  results: {
    fit_time_sec?: number;
    n_anomalies?: number;
    anomaly_fraction?: number;
    [key: string]: unknown;
  };
}

export interface HealthResponse {
  status: string;
  framework: string;
  version: string;
  dataset_loaded: boolean;
  dataset_name?: string | null;
  detection_ready: boolean;
  active_model: string;
  anomalies_detected: number | null;
  preprocessed_features?: number;
  /** True when the server requires X-API-Key. */
  auth_required?: boolean;
}

/**
 * Per-call transport options.
 *
 * ``signal`` lets a caller cancel a request; ``timeoutMs`` overrides the
 * per-endpoint budget.  Both were previously missing: a hung backend left the
 * UI spinning forever, and the ``AbortController`` that ``AppContext`` created
 * was never passed to ``fetch``, so "cancellation" was decorative.
 */
export interface ApiOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

/** Ordinary queries. */
const DEFAULT_TIMEOUT_MS = 30_000;
/**
 * Training, SHAP, benchmarks and dataset preprocessing are CPU-bound on the
 * server and legitimately take minutes.  A 30s budget would abort work the
 * backend had already committed to.
 */
const LONG_TIMEOUT_MS = 300_000;

/** An HTTP error with the status code and the server's own message. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function isApiError(err: unknown): err is ApiError {
  return err instanceof ApiError;
}

export function isAbortError(err: unknown): boolean {
  return (
    (err instanceof DOMException && err.name === "AbortError") ||
    (err instanceof Error && err.name === "AbortError")
  );
}

/**
 * FastAPI returns ``{"detail": "..."}`` for handled errors but
 * ``{"detail": [{"loc": ..., "msg": ...}]}`` for 422 validation failures.
 * ``String(array)`` renders as ``[object Object]``, which is how validation
 * errors used to reach the toast — so the array is flattened explicitly.
 */
function formatDetail(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object" || !("detail" in body)) return fallback;
  const detail = (body as { detail: unknown }).detail;

  if (typeof detail === "string") return detail || fallback;

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object" && "msg" in item) {
          const record = item as { msg: unknown; loc?: unknown };
          const where = Array.isArray(record.loc)
            ? record.loc.filter((part) => part !== "body").join(".")
            : "";
          return where ? `${where}: ${String(record.msg)}` : String(record.msg);
        }
        return JSON.stringify(item);
      })
      .filter(Boolean);
    if (messages.length > 0) return messages.join("; ");
  }

  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return fallback;
}

/** ``AbortSignal.timeout`` with a fallback for runtimes that lack it (jsdom). */
function timeoutSignal(timeoutMs: number): AbortSignal {
  if (typeof AbortSignal.timeout === "function") {
    return AbortSignal.timeout(timeoutMs);
  }
  const controller = new AbortController();
  setTimeout(() => controller.abort(new DOMException("Timeout", "AbortError")), timeoutMs);
  return controller.signal;
}

/** Combine the caller's signal with a timeout without leaking either. */
function combineSignals(
  timeoutMs: number,
  external?: AbortSignal
): { signal: AbortSignal; didTimeOut: () => boolean } {
  const timeout = timeoutSignal(timeoutMs);
  if (!external) {
    return { signal: timeout, didTimeOut: () => timeout.aborted };
  }

  if (typeof AbortSignal.any === "function") {
    return {
      signal: AbortSignal.any([external, timeout]),
      didTimeOut: () => timeout.aborted && !external.aborted,
    };
  }

  // Fallback for runtimes without AbortSignal.any (older Safari).
  const controller = new AbortController();
  const forward = () => controller.abort();
  external.addEventListener("abort", forward, { once: true });
  timeout.addEventListener("abort", forward, { once: true });
  return {
    signal: controller.signal,
    didTimeOut: () => timeout.aborted && !external.aborted,
  };
}

async function request<T>(
  endpoint: string,
  options?: RequestInit & ApiOptions
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const { signal, didTimeOut } = combineSignals(timeoutMs, options?.signal);

  let res: Response;
  try {
    res = await fetch(url, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
        ...options?.headers,
      },
      cache: "no-store",
      signal,
    });
  } catch (err: unknown) {
    if (didTimeOut()) {
      throw new ApiError(
        408,
        `Request timed out after ${Math.round(timeoutMs / 1000)}s. ` +
          "The operation may still be running on the server."
      );
    }
    if (isAbortError(err)) throw err;
    // A network failure (server down, DNS, CORS rejection) has no status code.
    console.error(`API network error on ${endpoint}:`, err);
    throw new ApiError(
      0,
      "Could not reach the SENTINEL API. Check that the backend is running."
    );
  }

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = formatDetail(body, `${res.status} ${res.statusText}`);
    if (res.status >= 500) {
      console.error(`API server error on ${endpoint} (${res.status}):`, detail);
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  // Health
  getHealth: (opts?: ApiOptions) =>
    request<HealthResponse>("/api/health", opts),

  // Dataset
  loadSample: (opts?: ApiOptions) =>
    request<{ success: boolean; dataset: DatasetSummary }>(
      "/api/dataset/sample",
      { method: "POST", ...opts }
    ),

  uploadCsv: async (file: File, opts?: ApiOptions) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<{ success: boolean; dataset: DatasetSummary }>(
      "/api/dataset/upload",
      { method: "POST", body: formData, timeoutMs: LONG_TIMEOUT_MS, ...opts }
    );
  },

  getDatasetSummary: (opts?: ApiOptions) =>
    request<DatasetSummary>("/api/dataset/summary", opts),

  preprocess: (
    config: { scaler_type: string; label_col?: string | null; id_cols?: string[] },
    opts?: ApiOptions
  ) =>
    request<{ success: boolean; preprocessing: PreprocessingSummary }>(
      "/api/dataset/preprocess",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
        timeoutMs: LONG_TIMEOUT_MS,
        ...opts,
      }
    ),

  resetState: (opts?: ApiOptions) =>
    request<{ success: boolean; message: string }>("/api/dataset/reset", {
      method: "POST",
      ...opts,
    }),

  // Detection
  getModels: (opts?: ApiOptions) =>
    request<{ models: Record<string, unknown> }>("/api/detection/models", opts),

  runDetection: (
    payload: {
      model_name: string;
      params?: Record<string, unknown>;
      threshold_method?: string;
      contamination?: number;
      quantile_val?: number;
      fixed_val?: number;
      use_ensemble?: boolean;
      ensemble_weights?: Record<string, number>;
    },
    opts?: ApiOptions
  ) =>
    request<{ success: boolean; overview: DetectionOverview }>(
      "/api/detection/run",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        timeoutMs: LONG_TIMEOUT_MS,
        ...opts,
      }
    ),

  getResults: (opts?: ApiOptions) =>
    request<DetectionOverview>("/api/detection/results", opts),

  getExperiments: (opts?: ApiOptions) =>
    request<{ experiments: ExperimentRecord[] }>(
      "/api/detection/experiments",
      opts
    ),

  // Anomalies & Investigation
  getAnomalies: (
    params: {
      search?: string;
      severity?: string[];
      min_score?: number;
      max_score?: number;
      anomalies_only?: boolean;
      sort_by?: string;
      ascending?: boolean;
      page?: number;
      page_size?: number;
    },
    opts?: ApiOptions
  ) => {
    const q = new URLSearchParams();
    if (params.search) q.append("search", params.search);
    if (params.severity && params.severity.length > 0) {
      params.severity.forEach((s) => q.append("severity", s));
    }
    if (params.min_score !== undefined) q.append("min_score", params.min_score.toString());
    if (params.max_score !== undefined) q.append("max_score", params.max_score.toString());
    if (params.anomalies_only !== undefined) q.append("anomalies_only", params.anomalies_only.toString());
    if (params.sort_by) q.append("sort_by", params.sort_by);
    if (params.ascending !== undefined) q.append("ascending", params.ascending.toString());
    if (params.page) q.append("page", params.page.toString());
    if (params.page_size) q.append("page_size", params.page_size.toString());

    return request<AnomalyListResponse>(`/api/anomalies?${q.toString()}`, opts);
  },

  investigateAnomaly: (rowId: number, opts?: ApiOptions) =>
    request<InvestigationDetail>(`/api/anomalies/${rowId}/investigate`, opts),

  // Explainability
  generateExplanations: (
    payload: { use_shap?: boolean; top_k?: number },
    opts?: ApiOptions
  ) =>
    request<{
      n_explained: number;
      top_driver: string;
      global_importance: GlobalImportanceEntry[];
    }>("/api/explainability/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      timeoutMs: LONG_TIMEOUT_MS,
      ...opts,
    }),

  getGlobalImportance: (opts?: ApiOptions) =>
    request<GlobalImportanceResponse>("/api/explainability/global", opts),

  getExplanationStability: (modelName?: string, opts?: ApiOptions) =>
    request<StabilityReport>(
      `/api/explainability/stability${
        modelName ? `?model_name=${encodeURIComponent(modelName)}` : ""
      }`,
      { timeoutMs: LONG_TIMEOUT_MS, ...opts }
    ),

  getFeatureDistribution: (feature: string, opts?: ApiOptions) =>
    request<{
      feature: string;
      normal_median: number;
      anomaly_median: number;
      separation: number;
      distribution: Array<{
        bin_start: number;
        bin_end: number;
        normal_count: number;
        anomaly_count: number;
      }>;
    }>(
      `/api/explainability/feature-distributions?feature=${encodeURIComponent(feature)}`,
      opts
    ),

  // Analytics
  getPca: (opts?: ApiOptions) =>
    request<PcaAnalyticsResponse>("/api/analytics/pca", opts),
  getCorrelations: (maxFeatures: number = 12, opts?: ApiOptions) =>
    request<{ features: string[]; matrix: number[][] }>(
      `/api/analytics/correlations?max_features=${maxFeatures}`,
      opts
    ),

  // Model Lab
  benchmarkModels: (contamination: number = 0.05, opts?: ApiOptions) =>
    request<ModelBenchmarkResponse>("/api/models/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contamination }),
      timeoutMs: LONG_TIMEOUT_MS,
      ...opts,
    }),

  // Evaluation — out-of-sample metrics
  getHoldoutMetrics: (
    testSize: number = 0.3,
    modelName?: string,
    opts?: ApiOptions
  ) => {
    const q = new URLSearchParams({ test_size: String(testSize) });
    if (modelName) q.append("model_name", modelName);
    return request<HoldoutMetrics>(`/api/evaluation/holdout?${q.toString()}`, {
      timeoutMs: LONG_TIMEOUT_MS,
      ...opts,
    });
  },

  getHoldoutComparison: (testSize: number = 0.3, opts?: ApiOptions) =>
    request<HoldoutComparison>(
      `/api/evaluation/holdout/all?test_size=${testSize}`,
      { timeoutMs: LONG_TIMEOUT_MS, ...opts }
    ),

  // Reports
  generateReport: (opts?: ApiOptions) =>
    request<AuditReport>("/api/reports/generate", {
      timeoutMs: LONG_TIMEOUT_MS,
      ...opts,
    }),

  // Export URLs
  exportPredictionsUrl: `${API_BASE}/api/export/predictions`,
  exportAnomaliesUrl: `${API_BASE}/api/export/anomalies`,
  exportExplanationsUrl: `${API_BASE}/api/export/explanations`,

  // XAUD 2.0 Endpoints
  getDatasetProfile: (opts?: ApiOptions) =>
    request<DatasetProfileResponse>("/api/dataset/profile", opts),

  getModelCapabilities: (opts?: ApiOptions) =>
    request<Record<string, ModelCapability>>("/api/models/capabilities", opts),

  getAvailableModels: (opts?: ApiOptions) =>
    request<{
      models: string[];
      capabilities: Record<string, ModelCapability>;
      default_params: Record<string, Record<string, unknown>>;
    }>("/api/models/available", opts),

  runEnsemble: (payload: EnsembleRunRequest, opts?: ApiOptions) =>
    request<EnsembleResponse>("/api/detection/ensemble", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      timeoutMs: LONG_TIMEOUT_MS,
      ...opts,
    }),

  evaluateDrift: (payload?: { psi_bins?: number }, opts?: ApiOptions) =>
    request<DriftReport>("/api/drift/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload ?? { psi_bins: 10 }),
      timeoutMs: LONG_TIMEOUT_MS,
      ...opts,
    }),

  getDriftStatus: (opts?: ApiOptions) =>
    request<DriftReport>("/api/drift/status", opts),

  runSyntheticBenchmark: (
    payload: {
      anomaly_type?: "point" | "multivariate" | "cluster" | "contextual";
      contamination?: number;
      difficulty?: "easy" | "medium" | "hard";
      severity?: number;
      models_to_test?: string[];
    },
    opts?: ApiOptions
  ) =>
    request<SyntheticBenchmarkResponse>("/api/benchmarks/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      timeoutMs: LONG_TIMEOUT_MS,
      ...opts,
    }),

  getBenchmarkStatus: (opts?: ApiOptions) =>
    request<SyntheticBenchmarkResponse>("/api/benchmarks/status", opts),

  getExperimentHistory: (opts?: ApiOptions) =>
    request<PersistentExperimentRecord[]>("/api/experiments/history", opts),

  getExperimentById: (id: string, opts?: ApiOptions) =>
    request<PersistentExperimentRecord>(`/api/experiments/${id}`, opts),

  // SENTINEL 3.0 Advanced Features
  // Conformal & Uncertainty
  calibrateConformal: (payload?: { alpha?: number; split_ratio?: number; group_column?: string }, opts?: ApiOptions) =>
    request<any>("/api/v2/conformal/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload ?? {}),
      ...opts,
    }),

  getConformalScores: (payload?: { alpha?: number; target_fdr?: number }, opts?: ApiOptions) =>
    request<any>("/api/v2/conformal/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload ?? {}),
      ...opts,
    }),

  // Advanced Explainability & Interactions
  getInteractions: (payload?: { top_n_pairs?: number; max_samples?: number }, opts?: ApiOptions) =>
    request<any>("/api/v2/explainability/interactions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload ?? { top_n_pairs: 5, max_samples: 50 }),
      ...opts,
    }),

  getAgreement: (record_index: number, top_k: number = 5, opts?: ApiOptions) =>
    request<any>("/api/v2/explainability/agreement", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record_index, top_k }),
      ...opts,
    }),

  getCohortExplanation: (cohort_name: string = "Flagged Anomalies", top_k: number = 5, opts?: ApiOptions) =>
    request<any>("/api/v2/explainability/cohort", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cohort_name, top_k }),
      ...opts,
    }),

  getDiverseCounterfactuals: (record_index: number, k_diverse: number = 3, max_features_modified: number = 3, opts?: ApiOptions) =>
    request<any>("/api/v2/explainability/diverse-counterfactuals", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record_index, k_diverse, max_features_modified }),
      ...opts,
    }),

  getInvestigationNarrative: (record_index: number, opts?: ApiOptions) =>
    request<any>("/api/v2/explainability/narrative", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record_index }),
      ...opts,
    }),

  // Human-in-the-Loop & Active Learning
  submitFeedback: (record_id: number, user_label: number, notes: string = "", opts?: ApiOptions) =>
    request<any>("/api/v2/feedback/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record_id, user_label, notes }),
      ...opts,
    }),

  getFeedbackSummary: (opts?: ApiOptions) =>
    request<any>("/api/v2/feedback/summary", opts),

  getActiveLearningQueue: (batch_size: number = 10, strategy: string = "hybrid", opts?: ApiOptions) =>
    request<any>("/api/v2/active-learning/queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ batch_size, strategy }),
      ...opts,
    }),

  retrainSemiSupervised: (held_out_ratio: number = 0.25, opts?: ApiOptions) =>
    request<any>("/api/v2/active-learning/retrain-semi-supervised", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ held_out_ratio }),
      ...opts,
    }),

  // Root-Cause & Robustness
  clusterRootCauses: (min_cluster_size: number = 3, opts?: ApiOptions) =>
    request<any>("/api/v2/root-cause/cluster", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ min_cluster_size }),
      ...opts,
    }),

  auditRobustness: (group_column?: string, test_noise: boolean = true, opts?: ApiOptions) =>
    request<any>("/api/v2/robustness/audit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group_column, test_noise }),
      ...opts,
    }),

  // Streaming & Time-Series
  ingestStreaming: (values: number[], opts?: ApiOptions) =>
    request<any>("/api/v2/streaming/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
      ...opts,
    }),

  recalibrateStreaming: (approved: boolean, opts?: ApiOptions) =>
    request<any>("/api/v2/streaming/recalibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved }),
      ...opts,
    }),

  // MLOps & Model Cards
  generateModelCard: (version: string = "3.0.0", author: string = "SENTINEL Auditor", opts?: ApiOptions) =>
    request<any>("/api/v2/mlops/model-card/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version, author }),
      ...opts,
    }),

  listRegistryModels: (opts?: ApiOptions) =>
    request<any>("/api/v2/mlops/registry/list", opts),

  saveModelToRegistry: (name?: string, version: string = "1.0.0", stage: string = "staging", opts?: ApiOptions) =>
    request<any>("/api/v2/mlops/registry/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, version, stage }),
      ...opts,
    }),

  promoteRegistryModel: (name: string, version: string, stage: string, opts?: ApiOptions) =>
    request<any>("/api/v2/mlops/registry/promote", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, version, stage }),
      ...opts,
    }),
};

