/**
 * View-level test for the drift monitor.
 *
 * It exercises the contract the backend actually returns (a *list* of
 * `feature_reports` with `drift_status`, not a map with `psi_severity`) plus the
 * two behaviours the audit flagged: a failed load is surfaced rather than
 * swallowed, and an unmount aborts the in-flight request.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { AppProvider } from "@/context/AppContext";
import { ToastProvider } from "@/components/ui/Toast";
import { DriftMonitorView } from "../DriftMonitorView";
import { api, type DriftReport } from "@/lib/api";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      ...actual.api,
      getHealth: vi.fn(),
      getDatasetSummary: vi.fn(),
      getResults: vi.fn(),
      getDriftStatus: vi.fn(),
      evaluateDrift: vi.fn(),
    },
  };
});

const REPORT: DriftReport = {
  overall_severity: "HIGH",
  mean_feature_psi: 0.31,
  n_features_evaluated: 3,
  features_high_drift: ["transaction_id"],
  features_moderate_drift: ["account_age_days"],
  feature_reports: [
    {
      feature: "transaction_id",
      type: "numerical",
      psi: 13.5571,
      ks_statistic: 1.0,
      p_value: 0.0,
      statistically_significant: true,
      drift_status: "HIGH",
      ref_mean: 1263.5,
      cur_mean: 1703.5,
    },
    {
      feature: "account_age_days",
      type: "numerical",
      psi: 0.11,
      ks_statistic: 0.08,
      p_value: 0.4,
      statistically_significant: false,
      drift_status: "MODERATE",
    },
    {
      feature: "segment",
      type: "categorical",
      js_distance: 0.02,
      drift_status: "STABLE",
    },
  ],
  actions: ["Investigate transaction_id upstream."],
};

function renderView() {
  return render(
    <ToastProvider>
      <AppProvider>
        <DriftMonitorView />
      </AppProvider>
    </ToastProvider>
  );
}

beforeEach(() => {
  vi.mocked(api.getHealth).mockResolvedValue({
    status: "healthy",
    framework: "XAUD",
    version: "2.1.0",
    dataset_loaded: false,
    detection_ready: false,
    active_model: "Isolation Forest",
    anomalies_detected: 0,
  });
  vi.mocked(api.getDatasetSummary).mockResolvedValue({ loaded: false });
  vi.mocked(api.getResults).mockResolvedValue({ ready: false } as never);
  vi.mocked(api.getDriftStatus).mockResolvedValue({ ready: false });
});

afterEach(() => {
  vi.mocked(api.getDriftStatus).mockReset();
});

describe("DriftMonitorView", () => {
  it("prompts for a dataset when none is loaded", async () => {
    renderView();
    expect(
      await screen.findByText(/no dataset loaded/i)
    ).toBeInTheDocument();
    expect(api.getDriftStatus).not.toHaveBeenCalled();
  });

  it("renders the backend's list-shaped feature reports", async () => {
    vi.mocked(api.getDatasetSummary).mockResolvedValue({
      loaded: true,
      n_rows: 880,
      n_preprocessed_features: 10,
    });
    vi.mocked(api.getDriftStatus).mockResolvedValue(REPORT);

    renderView();

    expect(await screen.findByText("transaction_id")).toBeInTheDocument();
    expect(screen.getByText("account_age_days")).toBeInTheDocument();
    // A categorical feature shows the JS distance, not a missing PSI.
    expect(screen.getByText("segment")).toBeInTheDocument();
    // The drift count is derived from the two lists, not a non-existent field.
    await waitFor(() =>
      expect(screen.getByText(/2 \/ 3/)).toBeInTheDocument()
    );
  });

  it("surfaces a failed load instead of silently rendering an empty panel", async () => {
    vi.mocked(api.getDatasetSummary).mockResolvedValue({ loaded: true });
    vi.mocked(api.getDriftStatus).mockRejectedValue(new Error("backend down"));

    renderView();

    expect(
      await screen.findByText(/could not load the drift report/i)
    ).toBeInTheDocument();
  });

  it("aborts the request when the view unmounts", async () => {
    vi.mocked(api.getDatasetSummary).mockResolvedValue({ loaded: true });
    let seenSignal: AbortSignal | undefined;
    vi.mocked(api.getDriftStatus).mockImplementation(async (opts) => {
      seenSignal = opts?.signal;
      return { ready: false };
    });

    const { unmount } = renderView();
    await waitFor(() => expect(seenSignal).toBeDefined());
    expect(seenSignal!.aborted).toBe(false);

    unmount();
    expect(seenSignal!.aborted).toBe(true);
  });
});
