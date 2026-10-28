import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

export default defineConfig([
  ...nextVitals,
  ...nextTs,
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    "coverage/**",
    "playwright-report/**",
    "test-results/**",
    "src/lib/api/schema.ts",
  ]),
  {
    rules: {
      // Shared frontend rule: never render raw HTML.
      "react/no-danger": "error",
      // The React Compiler is not enabled, and TanStack Table and Virtual trip this warning.
      "react-hooks/incompatible-library": "off",
    },
  },
]);
