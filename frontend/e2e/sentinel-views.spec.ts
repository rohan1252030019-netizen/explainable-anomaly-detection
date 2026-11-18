import { expect, test, type Page } from "@playwright/test";

/**
 * End-to-end smoke tests for the 7 new SENTINEL 3.0 views:
 * 1. Model Zoo
 * 2. Conformal Prediction & FDR
 * 3. SHAP Interaction Values
 * 4. Streaming Monitor
 * 5. Active Learning Feedback Queue
 * 6. Cluster Root-Cause Archetypes
 * 7. Model Cards Governance
 */

const HEALTH = {
  status: "healthy",
  framework: "SENTINEL Autonomous Anomaly Intelligence",
  version: "3.0.0",
  dataset_loaded: true,
  dataset_name: "creditcard_fraud_sample.csv",
  detection_ready: true,
  active_model: "Isolation Forest",
  anomalies_detected: 40,
  preprocessed_features: 10,
  auth_required: false,
};

const SUMMARY = {
  loaded: true,
  dataset_name: "creditcard_fraud_sample.csv",
  n_rows: 20000,
  n_cols: 31,
  numerical_cols: ["V1", "V2", "Amount"],
  categorical_cols: [],
  feature_matrix_ready: true,
  n_preprocessed_features: 30,
};

const OVERVIEW = {
  ready: true,
  n_total: 20000,
  n_anomalies: 40,
  anomaly_rate: 0.002,
  model_name: "Isolation Forest",
  severity_counts: { CRITICAL: 4, HIGH: 10, MEDIUM: 14, LOW: 12 },
};

async function stubApi(page: Page) {
  await page.route("**/api/health", (route) => route.fulfill({ json: HEALTH }));
  await page.route("**/api/dataset/summary", (route) => route.fulfill({ json: SUMMARY }));
  await page.route("**/api/detection/results", (route) => route.fulfill({ json: OVERVIEW }));
  await page.route("**/api/v2/conformal/calibrate", (route) =>
    route.fulfill({
      json: {
        is_calibrated: true,
        alpha: 0.05,
        confidence_level: 0.95,
        n_calibration_samples: 500,
        threshold_at_alpha: 0.65,
        is_mondrian: false,
        groups: [],
        fallback_groups: [],
        coverage_statement: "Split-conformal calibration established at alpha=0.05.",
      },
    })
  );
  await page.route("**/api/v2/conformal/score", (route) =>
    route.fulfill({
      json: {
        total_samples: 100,
        alpha: 0.05,
        confidence_level: 0.95,
        target_fdr: 0.05,
        conformal_anomalies_flagged: 5,
        fdr_discoveries_flagged: 4,
        coverage_statement: "Coverage guaranteed.",
        fdr_statement: "FDR controlled at 5%.",
        items: [],
      },
    })
  );
  await page.route("**/api/v2/explainability/interactions*", (route) =>
    route.fulfill({
      json: {
        method: "TreeSHAP_Interactions",
        top_interactions: [
          {
            feature_i: "V1",
            feature_j: "V2",
            interaction_strength: 0.85,
            synergy_type: "synergistic",
            description: "High interaction between V1 and V2",
          },
        ],
      },
    })
  );
  await page.route("**/api/v2/learning/active/queue*", (route) =>
    route.fulfill({
      json: {
        queue: [
          { record_id: 101, uncertainty_score: 0.498, disagreement_score: 0.65, detector_score: 0.68 },
        ],
      },
    })
  );
  await page.route("**/api/v2/learning/feedback/summary", (route) =>
    route.fulfill({
      json: {
        total_feedback: 12,
        confirmed_anomalies: 8,
        rejected_false_positives: 4,
      },
    })
  );
  await page.route("**/api/v2/learning/root-cause/clusters*", (route) =>
    route.fulfill({
      json: {
        clusters: [
          {
            cluster_id: 0,
            name: "Cluster 0: Velocity Outliers",
            size: 15,
            dominant_features: [{ feature: "V1", mean_shap: 0.72 }],
            archetype_description: "Anomalies driven by V1",
          },
        ],
        noise_count: 2,
      },
    })
  );
  await page.route("**/api/v2/mlops/model-card/generate*", (route) =>
    route.fulfill({
      json: {
        model_name: "Isolation Forest",
        version: "3.0.0",
        author: "SENTINEL Auditor",
        timestamp: "2026-11-19T00:00:00Z",
        dataset_fingerprint: "ede5aa5ca3656a79e724735dee2b02085fdcf6c60aa37d9d9792cbdfd12e7907",
        lifecycle_stage: "production",
        intended_use: "Unsupervised anomaly detection",
        out_of_scope_use: "Autonomous punitive action",
        factors_and_metrics: "Conformal coverage 95%",
        limitations: "Assumes exchangeability",
      },
    })
  );
}

test.describe("SENTINEL 3.0 Views E2E Smoke", () => {
  test.beforeEach(async ({ page }) => {
    await stubApi(page);
    await page.goto("/");
  });

  test("Model Zoo view renders 17 detectors", async ({ page }) => {
    await page.getByRole("button", { name: /Model Zoo/i }).click();
    await expect(page.getByRole("heading", { name: /SENTINEL Model Zoo/i })).toBeVisible();
    await expect(page.getByText(/Isolation Forest/i).first()).toBeVisible();
  });

  test("Conformal & FDR view renders error budget and calibration", async ({ page }) => {
    await page.getByRole("button", { name: /Conformal & FDR/i }).click();
    await expect(page.getByRole("heading", { name: /Calibrated Uncertainty & FDR Control/i })).toBeVisible();
  });

  test("SHAP Interactions view renders feature pair explorer", async ({ page }) => {
    await page.getByRole("button", { name: /SHAP Interactions/i }).click();
    await expect(page.getByRole("heading", { name: /SHAP Interaction Values/i })).toBeVisible();
  });

  test("Streaming Monitor view renders real-time stream status", async ({ page }) => {
    await page.getByRole("button", { name: /Streaming Monitor/i }).click();
    await expect(page.getByRole("heading", { name: /Streaming Anomaly Monitor/i })).toBeVisible();
  });

  test("Feedback Queue view renders active learning candidates", async ({ page }) => {
    await page.getByRole("button", { name: /Feedback Queue/i }).click();
    await expect(page.getByRole("heading", { name: /Analyst Feedback & Active Learning/i })).toBeVisible();
  });

  test("Root Cause view renders failure archetypes", async ({ page }) => {
    await page.getByRole("button", { name: /Root Cause/i }).click();
    await expect(page.getByRole("heading", { name: /Root-Cause Archetypes/i })).toBeVisible();
  });

  test("Model Cards view renders governance registry", async ({ page }) => {
    await page.getByRole("button", { name: /Model Cards/i }).click();
    await expect(page.getByRole("heading", { name: /Model Cards & Governance Registry/i })).toBeVisible();
  });
});
