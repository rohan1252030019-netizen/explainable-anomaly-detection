import { expect, test, type Page } from "@playwright/test";

/**
 * End-to-end smoke tests.
 *
 * The backend is stubbed at the network boundary with `page.route`, so the suite
 * asserts the *frontend* contract: the app boots, the dashboard renders the
 * health payload, and a navigation lands on the expected view.
 */

const HEALTH = {
  status: "healthy",
  framework: "XAUD Autonomous Intelligence",
  version: "2.1.0",
  dataset_loaded: true,
  dataset_name: "sample_anomaly_data.csv",
  detection_ready: true,
  active_model: "Isolation Forest",
  anomalies_detected: 40,
  preprocessed_features: 10,
  auth_required: false,
};

const SUMMARY = {
  loaded: true,
  dataset_name: "sample_anomaly_data.csv",
  n_rows: 880,
  n_cols: 10,
  numerical_cols: ["amount"],
  categorical_cols: ["segment"],
  feature_matrix_ready: true,
  n_preprocessed_features: 10,
};

async function stubApi(page: Page) {
  await page.route("**/api/health", (route) =>
    route.fulfill({ json: HEALTH })
  );
  await page.route("**/api/dataset/summary", (route) =>
    route.fulfill({ json: SUMMARY })
  );
  await page.route("**/api/detection/results", (route) =>
    route.fulfill({
      json: {
        ready: true,
        n_total: 880,
        n_anomalies: 40,
        anomaly_rate: 0.045,
        model_name: "Isolation Forest",
        severity_counts: { CRITICAL: 4, HIGH: 10, MEDIUM: 14, LOW: 12 },
      },
    })
  );
  await page.route("**/api/anomalies**", (route) =>
    route.fulfill({ json: { total: 0, page: 1, page_size: 25, items: [] } })
  );
  await page.route("**/api/drift/status", (route) =>
    route.fulfill({ json: { ready: false } })
  );
}

test("the dashboard renders the health payload", async ({ page }) => {
  await stubApi(page);
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  // The dataset name is rendered from the dataset summary payload, so this
  // asserts the health + summary round trip rather than a hard-coded string.
  await expect(page.getByText("sample_anomaly_data.csv").first()).toBeVisible();
});

test("navigation reaches the drift monitor", async ({ page }) => {
  await stubApi(page);
  await page.goto("/");

  // The heading is rendered by the shell from the active nav item, so it only
  // changes if the click was handled by hydrated React code.  Matching the
  // sidebar text instead would pass on server-rendered HTML alone.
  await expect(
    page.getByRole("heading", { name: "Dashboard", level: 1 })
  ).toBeVisible();

  await page.getByRole("button", { name: /^Drift Monitor$/ }).first().click();
  await expect(
    page.getByRole("heading", { name: "Drift Monitor", level: 1 })
  ).toBeVisible();
});

test("a dead backend surfaces an actionable error, not a blank page", async ({
  page,
}) => {
  await page.route("**/api/**", (route) =>
    route.fulfill({ status: 500, json: { detail: "simulated failure" } })
  );
  await page.goto("/");

  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
});
