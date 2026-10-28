import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    css: false,
    maxWorkers: 4,
    testTimeout: 15000,
    // Tests always use the real typed client against the MSW node server, never in-process mocks.
    env: { NEXT_PUBLIC_USE_MOCKS: "0" },
    restoreMocks: true,
  },
});
