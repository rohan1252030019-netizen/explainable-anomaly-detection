import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

export default defineConfig({
  // vitest ships its own copy of vite, so the plugin instance is structurally
  // identical but nominally from a different package.  The cast is a type-level
  // workaround only; the plugin runs the same code.
  plugins: [react()] as never,
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    // Playwright owns e2e/; vitest must not try to run those files.
    exclude: ["node_modules", ".next", "e2e/**"],
    restoreMocks: true,
    clearMocks: true,
  },
});
