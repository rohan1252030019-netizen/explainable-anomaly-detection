"use client";

import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  Database,
  Info,
  LineChart,
  RefreshCw,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { api, type PcaAnalyticsResponse } from "@/lib/api";
import { PcaScatterChart } from "@/components/charts/PcaScatterChart";
import { CorrelationHeatmap } from "@/components/charts/CorrelationHeatmap";
import { Button } from "@/components/ui/Button";
import { Card, SectionLabel } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonRows } from "@/components/ui/Feedback";
import { errorMessage } from "@/lib/errors";

interface AnalyticsData {
  pca: PcaAnalyticsResponse | null;
  corr: { features: string[]; matrix: number[][] } | null;
  loaded: boolean;
  pcaError: string | null;
  corrError: string | null;
}

const INITIAL: AnalyticsData = {
  pca: null,
  corr: null,
  loaded: false,
  pcaError: null,
  corrError: null,
};

export const AnalyticsView: React.FC = () => {
  const { overview, dataset, setActiveTab } = useApp();

  const [data, setData] = useState<AnalyticsData>(INITIAL);
  const [reloadToken, setReloadToken] = useState(0);

  const loaded = !!dataset?.loaded;
  const ready = !!overview?.ready;
  const matrixReady = dataset?.feature_matrix_ready === true;

  useEffect(() => {
    if (!loaded) return;
    let cancelled = false;

    Promise.allSettled([api.getPca(), api.getCorrelations(12)]).then(
      ([pcaResult, corrResult]) => {
        if (cancelled) return;

        setData({
          pca: pcaResult.status === "fulfilled" ? pcaResult.value : null,
          corr: corrResult.status === "fulfilled" ? corrResult.value : null,
          loaded: true,
          pcaError:
            pcaResult.status === "rejected"
              ? errorMessage(pcaResult.reason)
              : null,
          corrError:
            corrResult.status === "rejected"
              ? errorMessage(corrResult.reason)
              : null,
        });
      }
    );

    return () => {
      cancelled = true;
    };
  }, [loaded, ready, reloadToken]);

  if (!loaded) {
    return (
      <EmptyState
        icon={Database}
        title="No dataset to visualise"
        description="Load a CSV or the benchmark sample to inspect cluster separability, PCA projections and feature correlations."
        actionText="Open dataset"
        onAction={() => setActiveTab("dataset")}
      />
    );
  }

  const retry = () => {
    setData(INITIAL);
    setReloadToken((token) => token + 1);
  };

  const pcaVariance =
    data.pca?.variance_explained
      ?.slice(0, 2)
      .reduce((sum, value) => sum + value, 0) ?? null;

  return (
    <div className="space-y-5">
      {/* Context strip */}
      <Card className="flex flex-wrap items-center gap-x-8 gap-y-3 p-4">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="grid size-8 shrink-0 place-items-center rounded-lg border border-border bg-surface-muted text-primary">
            <LineChart size={15} aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <SectionLabel>Dataset</SectionLabel>
            <p className="truncate text-xs font-semibold text-foreground">
              {dataset?.dataset_name || "Untitled"}
            </p>
          </div>
        </div>

        <div className="hidden h-8 w-px bg-border sm:block" aria-hidden="true" />

        <div>
          <SectionLabel>PCA variance</SectionLabel>
          <p className="text-xs font-semibold tabular text-foreground">
            {pcaVariance !== null
              ? `${(pcaVariance * 100).toFixed(1)}% in two components`
              : "Not available"}
          </p>
        </div>

        <div className="hidden h-8 w-px bg-border sm:block" aria-hidden="true" />

        <div>
          <SectionLabel>Correlation scope</SectionLabel>
          <p className="text-xs font-semibold tabular text-foreground">
            {data.corr?.features?.length
              ? `${data.corr.features.length} numeric features`
              : "Not available"}
          </p>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <span
            className="rounded-full border border-border bg-surface-muted px-2.5 py-1 text-[10px] font-semibold text-muted-foreground"
            title={
              ready
                ? "Points are coloured by the active detector's output."
                : "Run detection to colour points by anomaly status."
            }
          >
            {ready ? "Detector result overlaid" : "Unclassified projection"}
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={retry}
            aria-label="Refresh analytics"
            title="Refresh analytics"
          >
            <RefreshCw size={13} aria-hidden="true" />
            Refresh
          </Button>
        </div>
      </Card>

      {!ready && (
        <div className="flex items-start gap-3 rounded-xl border border-info-line bg-info-soft p-3.5">
          <Info
            size={14}
            className="mt-0.5 shrink-0 text-info"
            aria-hidden="true"
          />
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Detection has not run for this dataset, so the projection is shown
            without anomaly labels. Run detection to overlay scores and
            severity.
          </p>
        </div>
      )}

      {!data.loaded ? (
        <div className="space-y-5">
          <Card className="p-6">
            <p className="mb-4 text-[10px] font-bold uppercase tracking-[0.14em] text-subtle-foreground">
              Projection
            </p>
            <SkeletonRows rows={5} />
            <p className="mt-4 text-center text-[10px] text-subtle-foreground" role="status">
              Computing PCA projection and correlation matrix…
            </p>
          </Card>
        </div>
      ) : (
        <div className="space-y-5">
          {data.pca ? (
            <Card className="p-5">
              <PcaScatterChart data={data.pca} />
            </Card>
          ) : data.pcaError ? (
            <Card
              role="alert"
              className="flex flex-col items-center gap-3 p-8 text-center"
            >
              <span className="grid size-9 place-items-center rounded-xl border border-danger-line bg-danger-soft text-danger">
                <AlertTriangle size={16} aria-hidden="true" />
              </span>
              <div>
                <p className="text-xs font-bold text-foreground">
                  Projection unavailable
                </p>
                <p className="mx-auto mt-1 max-w-lg break-words text-[11px] leading-relaxed text-muted-foreground">
                  {data.pcaError}
                </p>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2">
                {!matrixReady && (
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setActiveTab("dataset")}
                  >
                    <Database size={13} aria-hidden="true" />
                    Prepare features
                  </Button>
                )}
                <Button variant="ghost" size="sm" onClick={retry}>
                  <RefreshCw size={13} aria-hidden="true" />
                  Retry
                </Button>
              </div>
            </Card>
          ) : (
            <Card className="p-8 text-center text-xs text-subtle-foreground">
              No projection data was returned for this dataset.
            </Card>
          )}

          {data.corr && data.corr.features.length >= 2 ? (
            <Card className="p-5">
              <CorrelationHeatmap
                features={data.corr.features}
                matrix={data.corr.matrix}
              />
            </Card>
          ) : data.corrError ? (
            <Card
              role="alert"
              className="flex flex-col items-center gap-3 p-8 text-center"
            >
              <span className="grid size-9 place-items-center rounded-xl border border-danger-line bg-danger-soft text-danger">
                <AlertTriangle size={16} aria-hidden="true" />
              </span>
              <div>
                <p className="text-xs font-bold text-foreground">
                  Correlation matrix unavailable
                </p>
                <p className="mx-auto mt-1 max-w-lg break-words text-[11px] leading-relaxed text-muted-foreground">
                  {data.corrError}
                </p>
              </div>
              <Button variant="ghost" size="sm" onClick={retry}>
                <RefreshCw size={13} aria-hidden="true" />
                Retry
              </Button>
            </Card>
          ) : (
            <Card className="p-8 text-center">
              <p className="text-xs font-semibold text-foreground">
                Not enough numeric features
              </p>
              <p className="mx-auto mt-1 max-w-md text-[11px] leading-relaxed text-muted-foreground">
                The correlation matrix needs at least two numeric columns. This
                dataset does not provide them.
              </p>
            </Card>
          )}
        </div>
      )}
    </div>
  );
};
