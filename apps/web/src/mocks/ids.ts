/** Fixed ids of the mock backend. Constants only, so app code may import this file. */

/** A finished analysis of MOCK_REPO_URL (the demo analysis). */
export const MOCK_ANALYSIS_ID = "7d6c2f0e-3b8a-4c1d-9e5f-0a1b2c3d4e5f";
/** An analysis that failed in the mine stage. */
export const MOCK_FAILED_ANALYSIS_ID = "9b1e4a7c-5d2f-4e8a-b3c6-1f2e3d4c5b6a";
export const MOCK_REPOSITORY_ID = "3a5f8c2e-1d4b-4f6a-8e9c-7b0a1c2d3e4f";
/** Submitting this URL reuses MOCK_ANALYSIS_ID; any other valid URL starts a simulated run. */
export const MOCK_REPO_URL = "https://github.com/example/inventory-service";
export const MOCK_COMMIT_SHA = "8f3a1c5e9b2d4f6a0c7e1b3d5f7a9c2e4b6d8f0a";
export const MOCK_MODEL_VERSION = "ranker-2026.10.1";
/** File ids are MOCK_FIRST_FILE_ID + rank (0 is the highest priority). */
export const MOCK_FIRST_FILE_ID = 1001;
/** Advice for this file always fails. */
export const MOCK_ADVICE_FAILED_FILE_ID = 1003;
/** Advice is disabled for this file (as when no LLM key is configured). */
export const MOCK_ADVICE_DISABLED_FILE_ID = 1040;
