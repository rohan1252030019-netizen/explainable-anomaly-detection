import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { AppProvider } from "@/context/AppContext";
import { ToastProvider } from "@/components/ui/Toast";
import { ModelZooView } from "../ModelZooView";
import { ConformalView } from "../ConformalView";
import { InteractionsView } from "../InteractionsView";
import { StreamingView } from "../StreamingView";
import { FeedbackQueueView } from "../FeedbackQueueView";
import { RootCauseView } from "../RootCauseView";
import { ModelCardsView } from "../ModelCardsView";

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      ...actual.api,
      getHealth: vi.fn().mockResolvedValue({ status: "healthy", framework: "fastapi", version: "3.0.0" }),
      getDatasetSummary: vi.fn().mockResolvedValue({ loaded: true, n_rows: 500, n_preprocessed_features: 10 }),
      getResults: vi.fn().mockResolvedValue({ ready: true, model_name: "Isolation Forest", n_anomalies: 25 }),
      calibrateConformal: vi.fn().mockResolvedValue({ status: "success", alpha: 0.05, n_calibration: 150 }),
      getConformalScores: vi.fn().mockResolvedValue({ p_values: [0.01, 0.45], fdr_significant: [true, false] }),
      getInteractions: vi.fn().mockResolvedValue({ top_interactions: [] }),
      ingestStreaming: vi.fn().mockResolvedValue({ score: 0.22, is_anomaly: false, psi: 0.04 }),
      recalibrateStreaming: vi.fn().mockResolvedValue({ status: "recalibrated" }),
      getActiveLearningQueue: vi.fn().mockResolvedValue({ queue: [] }),
      getFeedbackSummary: vi.fn().mockResolvedValue({ total_reviewed: 0, confirmed_anomalies: 0, rejected_fp: 0 }),
      clusterRootCauses: vi.fn().mockResolvedValue({ clusters: [], noise_count: 0 }),
      generateModelCard: vi.fn().mockResolvedValue({ model_name: "Isolation Forest", version: "3.0.0" }),
    },
  };
});

function wrap(ui: React.ReactElement) {
  return render(
    <ToastProvider>
      <AppProvider>{ui}</AppProvider>
    </ToastProvider>
  );
}

describe("SENTINEL Phase 8 Frontend Views", () => {
  it("renders ModelZooView and lists available detectors", () => {
    wrap(<ModelZooView />);
    expect(screen.getByText(/SENTINEL Model Zoo/i)).toBeInTheDocument();
    expect(screen.getByText("Isolation Forest")).toBeInTheDocument();
  });

  it("renders ConformalView with calibration and error budget", () => {
    wrap(<ConformalView />);
    expect(screen.getByText(/Calibrated Uncertainty & FDR Control/i)).toBeInTheDocument();
  });

  it("renders InteractionsView", () => {
    wrap(<InteractionsView />);
    expect(screen.getByText(/SHAP Feature Interactions|Analysis Required/i)).toBeInTheDocument();
  });

  it("renders StreamingView with monitor controls", () => {
    wrap(<StreamingView />);
    expect(screen.getByText(/Streaming Monitor & Concept Drift/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Ingest Streaming Event/i })).toBeInTheDocument();
  });

  it("renders FeedbackQueueView with active learning triage queue", () => {
    wrap(<FeedbackQueueView />);
    expect(screen.getByText(/Active Learning & Feedback Queue|Analysis Required/i)).toBeInTheDocument();
  });

  it("renders RootCauseView with failure archetypes", () => {
    wrap(<RootCauseView />);
    expect(screen.getByText(/Root-Cause Archetype Clusters|Analysis Required/i)).toBeInTheDocument();
  });

  it("renders ModelCardsView with governance controls", () => {
    wrap(<ModelCardsView />);
    expect(screen.getByText(/Model Cards & Governance Registry/i)).toBeInTheDocument();
  });
});
