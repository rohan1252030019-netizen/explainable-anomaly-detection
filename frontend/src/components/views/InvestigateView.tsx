"use client";

import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Info,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Target,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import {
  api,
  type Counterfactual,
  type ExplanationVerification,
  type InvestigationDetail,
} from "@/lib/api";
import { ScoreGauge } from "@/components/ui/ScoreGauge";
import { SeverityBadge, severityStyle } from "@/components/ui/Badge";
import { FeatureContributionChart } from "@/components/charts/FeatureContributionChart";
import { ObservedVsTypicalRange } from "@/components/charts/ObservedVsTypicalRange";
import { Button } from "@/components/ui/Button";
import { Card, SectionLabel } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { Tabs } from "@/components/ui/Tabs";
import { SkeletonRows } from "@/components/ui/Feedback";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

type DetailTab = "features" | "comparison" | "raw";

const IGNORED_RAW_KEYS = ["anomaly_score", "normalized_score", "prediction"];

/* ------------------------------------------------------------------ *
 * Counterfactual recommendation
 *
 * The attribution above answers "what was unusual". This answers "what
 * would have to be different", which is the question an analyst actually
 * has. The quality label is shown because a nearest-neighbour match can
 * still score well above the detection threshold, and presenting that as
 * a fix would be misleading.
 * ------------------------------------------------------------------ */

const QUALITY_STYLE: Record<
  string,
  {
    hex: string;
    label: string;
    icon: React.ComponentType<{ size?: number; className?: string }>;
  }
> = {
  STRONG: { hex: "var(--success)", label: "Clears the bar", icon: ShieldCheck },
  PARTIAL: {
    hex: "var(--caution)",
    label: "Helps, not enough",
    icon: ShieldAlert,
  },
  WEAK: { hex: "var(--warning)", label: "Poor match", icon: ShieldAlert },
  UNVERIFIED: { hex: "var(--subtle-foreground)", label: "Unverified", icon: Info },
  NONE: { hex: "var(--subtle-foreground)", label: "None", icon: Info },
};

const CounterfactualPanel: React.FC<{
  counterfactual?: Counterfactual | null;
  isAnomaly: boolean;
}> = ({ counterfactual, isAnomaly }) => {
  if (!counterfactual) {
    return (
      <Card className="p-5">
        <div className="flex items-start gap-3">
          <span className="grid size-8 shrink-0 place-items-center rounded-lg border border-border bg-surface-muted text-subtle-foreground">
            <Target size={15} aria-hidden="true" />
          </span>
          <div>
            <SectionLabel>What would fix this record</SectionLabel>
            <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
              No counterfactual was returned for this observation. The feature
              evidence and raw values are still available below.
            </p>
          </div>
        </div>
      </Card>
    );
  }

  const quality = counterfactual.quality
    ? (QUALITY_STYLE[counterfactual.quality.label] ?? QUALITY_STYLE.NONE)
    : QUALITY_STYLE.NONE;
  const QualityIcon = quality.icon;

  if (!counterfactual.found) {
    return (
      <Card className="p-5">
        <div className="flex items-start gap-3">
          <span className="grid size-8 shrink-0 place-items-center rounded-lg border border-border bg-surface-muted text-subtle-foreground">
            <Target size={15} aria-hidden="true" />
          </span>
          <div>
            <SectionLabel>What would fix this record</SectionLabel>
            <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
              {counterfactual.reason ??
                (isAnomaly
                  ? "No nearby non-flagged record could be located for this row."
                  : "This record is not flagged, so no corrective change is required.")}
            </p>
            <p className="mt-1 text-[10px] leading-relaxed text-subtle-foreground">
              A missing recommendation is a valid result for an outlier that sits
              far from every normal prototype.
            </p>
          </div>
        </div>
      </Card>
    );
  }

  const changes = counterfactual.changes ?? [];

  return (
    <Card>
      <div className="border-b border-border p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Target size={13} className="text-primary" aria-hidden="true" />
              <SectionLabel>What would fix this record</SectionLabel>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-foreground">
              {counterfactual.text}
            </p>
          </div>

          <span
            className="inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold"
            style={{
              color: quality.hex,
              borderColor: quality.hex,
            }}
            title={counterfactual.quality?.detail}
          >
            <QualityIcon size={11} aria-hidden="true" />
            {quality.label}
          </span>
        </div>

        {counterfactual.score_before !== null &&
          counterfactual.score_before !== undefined && (
            <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface-muted p-3">
              <SectionLabel>Score effect</SectionLabel>
              <span className="text-sm font-extrabold tabular-nums text-danger">
                {counterfactual.score_before.toFixed(3)}
              </span>
              <ArrowRight
                size={13}
                className="text-subtle-foreground"
                aria-hidden="true"
              />
              <span className="text-sm font-extrabold tabular-nums text-success">
                {counterfactual.score_after?.toFixed(3) ?? "—"}
              </span>
              {counterfactual.score_threshold !== null &&
                counterfactual.score_threshold !== undefined && (
                  <span className="text-[11px] text-subtle-foreground">
                    detection threshold{" "}
                    {counterfactual.score_threshold.toFixed(3)}
                  </span>
                )}
              {typeof counterfactual.score_reduction === "number" && (
                <span className="ml-auto rounded bg-surface px-2 py-0.5 text-[10px] font-semibold tabular text-muted-foreground">
                  −{counterfactual.score_reduction.toFixed(3)} score
                </span>
              )}
              {typeof counterfactual.intervention_cost === "number" && (
                <span className="rounded bg-surface px-2 py-0.5 text-[10px] font-semibold tabular text-primary border border-primary-line">
                  Intervention Cost: {counterfactual.intervention_cost.toFixed(3)}
                </span>
              )}
            </div>
          )}
      </div>

      {changes.length > 0 && (
        <div className="p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <SectionLabel>Fields to change</SectionLabel>
            {counterfactual.immutable_features_enforced && counterfactual.immutable_features_enforced.length > 0 && (
              <span className="text-[10px] text-subtle-foreground font-mono">
                Immutable features locked: {counterfactual.immutable_features_enforced.join(", ")}
              </span>
            )}
          </div>
          <ul className="mt-3 space-y-2">
            {changes.map((change) => {
              const increasing = change.direction === "increase";
              const Icon = increasing ? ArrowUpRight : ArrowDownRight;
              const shown =
                change.suggested_value !== undefined &&
                change.suggested_value !== null
                  ? change.suggested_value
                  : change.scaled_to;
              const base =
                change.observed_value !== undefined &&
                change.observed_value !== null
                  ? change.observed_value
                  : change.scaled_from;

              return (
                <li
                  key={change.feature}
                  className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-border bg-surface-muted px-3 py-2"
                >
                  <span className="w-44 shrink-0 truncate text-xs font-semibold text-foreground">
                    {change.feature}
                  </span>
                  <span className="flex items-center gap-1.5 font-mono text-xs tabular-nums text-muted-foreground">
                    {Number(base).toFixed(2)}
                    <Icon
                      size={12}
                      className={increasing ? "text-warning" : "text-info"}
                      aria-hidden="true"
                    />
                    <span className="font-bold text-foreground">
                      {Number(shown).toFixed(2)}
                    </span>
                  </span>
                  {change.change_pct !== null &&
                    change.change_pct !== undefined && (
                      <span className="text-[11px] tabular-nums text-subtle-foreground">
                        {change.change_pct > 0 ? "+" : ""}
                        {change.change_pct.toFixed(1)}%
                      </span>
                    )}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {counterfactual.quality?.detail && (
        <p className="border-t border-border px-5 py-3 text-[11px] leading-relaxed text-muted-foreground">
          {counterfactual.quality.detail}
        </p>
      )}
      {counterfactual.caveat && (
        <p className="border-t border-border px-5 py-3 text-[11px] leading-relaxed text-subtle-foreground">
          {counterfactual.caveat}
        </p>
      )}
    </Card>
  );
};

/* ------------------------------------------------------------------ *
 * Verification panel
 * ------------------------------------------------------------------ */

const VerificationPanel: React.FC<{
  verification?: ExplanationVerification | null;
}> = ({ verification }) => {
  if (!verification || verification.checks_run === 0) {
    return (
      <Card className="p-5">
        <div className="flex items-start gap-3">
          <span className="grid size-8 shrink-0 place-items-center rounded-lg border border-border bg-surface-muted text-subtle-foreground">
            <ShieldCheck size={15} aria-hidden="true" />
          </span>
          <div>
            <SectionLabel>Explanation check</SectionLabel>
            <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">
              {verification?.summary ||
                "No behavioural verification was run for this record, so the explanation is unverified rather than endorsed."}
            </p>
            <p className="mt-1 text-[10px] text-subtle-foreground">
              The feature evidence below still shows why the model scored this
              row, independent of the verification result.
            </p>
          </div>
        </div>
      </Card>
    );
  }

  const tone =
    verification.overall === "VERIFIED" ||
    verification.overall === "BEHAVIOURALLY FAITHFUL"
      ? "var(--success)"
      : verification.overall === "PARTIALLY VERIFIED"
        ? "var(--caution)"
        : verification.overall === "UNKNOWN"
          ? "var(--subtle-foreground)"
          : "var(--danger)";

  const faithfulness = verification.faithfulness;

  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-3 p-5">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {verification.overall === "UNVERIFIED" ? (
              <ShieldAlert
                size={13}
                className="text-danger"
                aria-hidden="true"
              />
            ) : (
              <ShieldCheck
                size={13}
                className="text-primary"
                aria-hidden="true"
              />
            )}
            <SectionLabel>Explanation check</SectionLabel>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-foreground">
            {verification.summary}
          </p>
        </div>
        <span
          className="shrink-0 rounded-full border px-2.5 py-1 text-[10px] font-bold"
          style={{ color: tone, borderColor: tone }}
        >
          {verification.overall}
        </span>
      </div>

      {faithfulness && faithfulness.deletion_auc_explained !== null && (
        <div className="grid grid-cols-1 gap-px border-t border-border bg-border sm:grid-cols-2">
          <div className="bg-surface p-4">
            <SectionLabel>Removal AUC — explained order</SectionLabel>
            <p className="mt-1 text-lg font-extrabold tabular-nums text-success">
              {faithfulness.deletion_auc_explained.toFixed(3)}
            </p>
            <p className="mt-0.5 text-[10px] text-subtle-foreground">
              Lower is better: the score should collapse as the named features
              are removed.
            </p>
          </div>
          <div className="bg-surface p-4">
            <SectionLabel>Removal AUC — random order</SectionLabel>
            <p className="mt-1 text-lg font-extrabold tabular-nums text-foreground">
              {faithfulness.deletion_auc_random?.toFixed(3) ?? "—"}
              {typeof faithfulness.deletion_auc_random_std === "number" && (
                <span className="ml-1 text-[10px] font-medium text-subtle-foreground">
                  ± {faithfulness.deletion_auc_random_std.toFixed(3)}
                </span>
              )}
            </p>
            <p className="mt-0.5 text-[10px] text-subtle-foreground">
              The baseline any explanation has to beat to be worth anything.
            </p>
          </div>
        </div>
      )}

      {(faithfulness?.n_features !== undefined ||
        verification.additivity?.checked) && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-border px-5 py-3 text-[10px] text-subtle-foreground">
          {faithfulness?.n_features !== undefined && (
            <span>
              Ablated{" "}
              <span className="font-semibold tabular text-muted-foreground">
                {faithfulness.n_features}
              </span>{" "}
              features
            </span>
          )}
          {verification.additivity?.checked && (
            <span className="inline-flex items-center gap-1">
              {verification.additivity.passed ? (
                <CheckCircle2
                  size={11}
                  className="text-success"
                  aria-hidden="true"
                />
              ) : (
                <AlertTriangle
                  size={11}
                  className="text-caution"
                  aria-hidden="true"
                />
              )}
              Additivity check{" "}
              {verification.additivity.passed ? "passed" : "did not pass"}
              {typeof verification.additivity.max_abs_error === "number" && (
                <span className="tabular">
                  {" "}
                  (max error{" "}
                  {verification.additivity.max_abs_error.toExponential(2)})
                </span>
              )}
            </span>
          )}
        </div>
      )}
    </Card>
  );
};

/* ------------------------------------------------------------------ *
 * Screen
 * ------------------------------------------------------------------ */

export const InvestigateView: React.FC = () => {
  const { dataset, overview, setActiveTab } = useApp();

  const [rowId, setRowId] = useState<number>(() => {
    if (typeof window !== "undefined") {
      const saved =
        localStorage.getItem("sentinel-selected-anomaly") ||
        localStorage.getItem("xaud-selected-anomaly");
      const parsed = saved === null ? 0 : Number(saved);
      if (Number.isInteger(parsed) && parsed >= 0) return parsed;
    }
    return 0;
  });
  const [rowDraft, setRowDraft] = useState(() => String(rowId));

  const [detail, setDetail] = useState<InvestigationDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rowError, setRowError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [tab, setTab] = useState<DetailTab>("features");

  const ready = !!overview?.ready;
  const maxRow =
    overview?.n_total !== undefined ? Math.max(0, overview.n_total - 1) : null;

  useEffect(() => {
    if (!ready) return;
    let cancelled = false;

    api
      .investigateAnomaly(rowId)
      .then((result) => {
        if (cancelled) return;
        setError(null);
        setDetail(result);
        setLoading(false);
        localStorage.setItem("sentinel-selected-anomaly", String(rowId));
        localStorage.setItem("xaud-selected-anomaly", String(rowId));
      })
      .catch((err) => {
        if (cancelled) return;
        setError(errorMessage(err));
        setDetail(null);
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [rowId, ready, reloadToken]);

  const goToRow = (next: number) => {
    const bounded =
      maxRow === null ? Math.max(0, next) : Math.min(maxRow, Math.max(0, next));
    setRowError(null);
    if (bounded === rowId) {
      setRowDraft(String(rowId));
      return;
    }
    setDetail(null);
    setError(null);
    setLoading(true);
    setRowDraft(String(bounded));
    setRowId(bounded);
  };

  const commitRowDraft = () => {
    const parsed = Number(rowDraft);
    if (
      !Number.isInteger(parsed) ||
      parsed < 0 ||
      (maxRow !== null && parsed > maxRow)
    ) {
      setRowError(
        maxRow !== null
          ? `Enter a whole row index between 0 and ${maxRow}.`
          : "Enter a whole row index of 0 or greater."
      );
      setRowDraft(String(rowId));
      return;
    }
    setRowError(null);
    goToRow(parsed);
  };

  if (!ready) {
    return (
      <EmptyState
        icon={Search}
        title="No observations to investigate"
        description={
          dataset?.loaded
            ? "Run detection first. The investigation workspace opens any row by its original dataset index."
            : "Load a dataset and run detection to inspect a record's evidence, counterfactual and explanation checks."
        }
        actionText={dataset?.loaded ? "Configure detection" : "Open dataset"}
        onAction={() =>
          setActiveTab(dataset?.loaded ? "detection" : "dataset")
        }
      />
    );
  }

  const outOfBounds =
    error !== null &&
    (error.toLowerCase().includes("out of bounds") ||
      error.toLowerCase().includes("not found"));

  return (
    <div className="space-y-4">
      {/* Observation navigator */}
      <Card className="p-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <button
            type="button"
            onClick={() => setActiveTab("anomalies")}
            className="inline-flex w-fit items-center gap-2 text-xs font-semibold text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft size={14} aria-hidden="true" />
            Back to anomalies table
          </button>

          <form
            onSubmit={(event) => {
              event.preventDefault();
              commitRowDraft();
            }}
            className="flex flex-wrap items-center gap-2"
          >
            <label htmlFor="obs-index" className="text-[11px] text-muted-foreground">
              Observation
            </label>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => goToRow(rowId - 1)}
                disabled={rowId <= 0}
                aria-label="Previous observation"
              >
                <ChevronLeft size={14} aria-hidden="true" />
              </Button>
              <input
                id="obs-index"
                type="number"
                inputMode="numeric"
                min={0}
                max={maxRow ?? undefined}
                value={rowDraft}
                onChange={(event) => setRowDraft(event.target.value)}
                onBlur={commitRowDraft}
                aria-invalid={rowError !== null}
                aria-describedby={rowError ? "obs-index-error" : undefined}
                className={cn(
                  "field field-mono h-8 w-20 text-center font-bold",
                  rowError && "border-danger"
                )}
              />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => goToRow(rowId + 1)}
                disabled={maxRow !== null && rowId >= maxRow}
                aria-label="Next observation"
              >
                <ChevronRight size={14} aria-hidden="true" />
              </Button>
              <Button variant="secondary" size="sm" type="submit">
                Go
              </Button>
            </div>
            {maxRow !== null && (
              <span className="text-[10px] text-subtle-foreground">
                of {maxRow.toLocaleString()}
              </span>
            )}
          </form>
        </div>
        {rowError && (
          <p
            id="obs-index-error"
            role="alert"
            className="mt-2 text-[10px] font-medium text-danger sm:text-right"
          >
            {rowError}
          </p>
        )}
      </Card>

      <div aria-busy={loading}>
        {loading || (!detail && !error) ? (
          <div className="space-y-4">
            <p
              role="status"
              className="px-1 text-[11px] text-subtle-foreground"
            >
              Loading evidence for observation #{rowId}…
            </p>
            <Card className="p-8">
              <SkeletonRows rows={4} />
            </Card>
            <Card className="p-6">
              <SkeletonRows rows={3} />
            </Card>
          </div>
        ) : error ? (
          <Card
            role="alert"
            className="flex flex-col items-center gap-3 p-10 text-center"
          >
            <span className="grid size-10 place-items-center rounded-xl border border-danger-line bg-danger-soft text-danger">
              <AlertTriangle size={18} aria-hidden="true" />
            </span>
            <div>
              <p className="text-sm font-bold text-foreground">
                {outOfBounds
                  ? `Observation #${rowId} is outside the dataset`
                  : `Could not load observation #${rowId}`}
              </p>
              <p className="mx-auto mt-1 max-w-lg break-words text-xs leading-relaxed text-muted-foreground">
                {error}
              </p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2">
              {!outOfBounds && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => {
                    setError(null);
                    setDetail(null);
                    setLoading(true);
                    setReloadToken((token) => token + 1);
                  }}
                >
                  <RefreshCw size={13} aria-hidden="true" />
                  Retry
                </Button>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setActiveTab("anomalies")}
              >
                Back to anomalies
              </Button>
            </div>
          </Card>
        ) : detail ? (
          <div className="space-y-4">
            {/* Verdict */}
            <Card
              className="overflow-hidden p-6"
              style={{
                borderLeftWidth: 4,
                borderLeftColor: severityStyle(detail.severity).hex,
              }}
            >
              <div className="grid grid-cols-1 items-center gap-6 md:grid-cols-[15rem_minmax(0,1fr)]">
                <div className="flex flex-col items-center border-b border-border pb-5 md:border-b-0 md:border-r md:pb-0 md:pr-6">
                  <SeverityBadge severity={detail.severity} className="mb-3" />
                  <ScoreGauge
                    score={detail.normalized_score}
                    severity={detail.severity}
                    threshold={detail.threshold}
                  />
                  <p className="mt-2 text-[11px] text-muted-foreground">
                    Observation #{detail.row_id}
                    {detail.rank !== null && detail.rank !== undefined
                      ? ` · rank #${detail.rank}`
                      : ""}
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Sparkles size={13} className="text-primary" aria-hidden="true" />
                    <SectionLabel>Transparent explainability</SectionLabel>
                  </div>
                  <h2 className="text-base font-extrabold tracking-tight text-foreground">
                    Why was this record flagged as anomalous?
                  </h2>
                  <p className="panel-inset p-4 text-xs leading-relaxed text-muted-foreground">
                    {detail.explanation_text ||
                      "Record exhibits statistical deviation from the baseline normal cluster."}
                  </p>
                  {detail.confidence_note && (
                    <p className="flex items-start gap-2 text-[11px] leading-relaxed text-muted-foreground">
                      <Info
                        size={13}
                        className="mt-px shrink-0 text-primary"
                        aria-hidden="true"
                      />
                      {detail.confidence_note}
                    </p>
                  )}
                  <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-border pt-3 text-[10px] text-subtle-foreground">
                    <span>
                      Detector ·{" "}
                      <span className="font-semibold text-muted-foreground">
                        {detail.model_name}
                      </span>
                    </span>
                    <span>
                      Threshold ·{" "}
                      <span className="font-semibold tabular text-muted-foreground">
                        {detail.threshold.toFixed(4)}
                      </span>
                    </span>
                    <span>
                      Status ·{" "}
                      <span
                        className={cn(
                          "font-semibold",
                          detail.is_anomaly ? "text-danger" : "text-success"
                        )}
                      >
                        {detail.is_anomaly ? "Flagged" : "Not flagged"}
                      </span>
                    </span>
                  </div>
                </div>
              </div>
            </Card>

            {/* Actionable recommendation and verification, above the raw
                detail tabs: these answer "what now" and "can I trust this". */}
            <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-2">
              <CounterfactualPanel
                counterfactual={detail.counterfactual}
                isAnomaly={detail.is_anomaly}
              />
              <VerificationPanel verification={detail.verification} />
            </div>

            {/* Detail tabs */}
            <Card className="overflow-hidden">
              <Tabs
                value={tab}
                onChange={setTab}
                tabs={[
                  { id: "features", label: "Feature contributions" },
                  {
                    id: "comparison",
                    label: "Observed vs baseline",
                    count: detail.feature_comparison?.length,
                  },
                  { id: "raw", label: "Raw record attributes" },
                ]}
              />

              <div className="min-h-[16rem] p-5">
                {tab === "features" && (
                  <FeatureContributionChart features={detail.top_features} />
                )}
                {tab === "comparison" && (
                  <ObservedVsTypicalRange items={detail.feature_comparison} />
                )}
                {tab === "raw" && (
                  <div className="space-y-3">
                    <div className="flex flex-wrap items-baseline justify-between gap-2">
                      <SectionLabel>
                        Observation #{detail.row_id} attribute profile
                      </SectionLabel>
                      <span className="text-[10px] text-subtle-foreground">
                        {Object.keys(detail.raw_record).length} fields
                      </span>
                    </div>
                    <dl className="grid grid-cols-2 gap-2 font-mono text-xs sm:grid-cols-3 md:grid-cols-4">
                      {Object.entries(detail.raw_record)
                        .filter(([key]) => !IGNORED_RAW_KEYS.includes(key))
                        .map(([key, value]) => (
                          <div
                            key={key}
                            className="panel-inset min-w-0 truncate p-2.5"
                            title={`${key}: ${value}`}
                          >
                            <dt className="block truncate font-sans text-[10px] uppercase text-subtle-foreground">
                              {key}
                            </dt>
                            <dd className="block truncate font-bold text-foreground">
                              {value !== null ? String(value) : "null"}
                            </dd>
                          </div>
                        ))}
                    </dl>
                  </div>
                )}
              </div>
            </Card>
          </div>
        ) : null}
      </div>
    </div>
  );
};
