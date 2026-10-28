/** The MSW node server for tests (started in vitest.setup.ts). Never import it from app code. */
import { setupServer } from "msw/node";

import { handlers } from "./handlers";

export const server = setupServer(...handlers);
