import { defineConfig, devices } from "@playwright/test";

const PORT = Number(process.env.E2E_PORT ?? 3210);

export default defineConfig({
  testDir: "./e2e",
  testMatch: /.*\.spec\.ts/,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "line" : "list",
  timeout: 60_000,
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  // A production build, not `next dev`: the dev server's origin checks block
  // the HMR socket when the page is loaded from 127.0.0.1, and in that state
  // the app never hydrates — clicks are no-ops and the mount effect never
  // fires, which makes a dev-server E2E suite assert server-rendered HTML only.
  // `next build && next start` exercises the same bundle users receive.
  webServer: {
    command: `npm run build && npm run start -- --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: false,
    timeout: 300_000,
    env: { NEXT_PUBLIC_API_URL: `http://127.0.0.1:${PORT}` },
  },
});
