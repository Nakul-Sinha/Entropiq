// Generates src/lib/api/schema.ts from the API's OpenAPI document.
//
//   node scripts/api-schema.mjs --write   regenerate the file
//   node scripts/api-schema.mjs --check   exit 1 when the file differs from the spec
//
// The spec is contracts/openapi.json at the repository root (or OPENAPI_SPEC). Until it exists the
// hand-written schema.ts is kept and both modes exit 0.
import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import openapiTS, { astToString, COMMENT_HEADER } from "openapi-typescript";
import * as prettier from "prettier";

const mode = process.argv[2];
if (process.argv.length !== 3 || (mode !== "--write" && mode !== "--check")) {
  console.error("usage: node scripts/api-schema.mjs --write | --check");
  process.exit(2);
}

const specUrl = process.env.OPENAPI_SPEC
  ? pathToFileURL(resolve(process.cwd(), process.env.OPENAPI_SPEC))
  : new URL("../../../contracts/openapi.json", import.meta.url);
const outPath = fileURLToPath(new URL("../src/lib/api/schema.ts", import.meta.url));

if (!existsSync(fileURLToPath(specUrl))) {
  console.log(
    "contracts/openapi.json does not exist yet; keeping the hand-written src/lib/api/schema.ts",
  );
  process.exit(0);
}

const ast = await openapiTS(specUrl);
const source = COMMENT_HEADER + astToString(ast);
const config = (await prettier.resolveConfig(outPath)) ?? {};
const formatted = await prettier.format(source, { ...config, filepath: outPath });

if (mode === "--write") {
  await writeFile(outPath, formatted, "utf8");
  console.log("wrote src/lib/api/schema.ts");
} else {
  const current = existsSync(outPath) ? await readFile(outPath, "utf8") : "";
  if (current.replace(/\r\n/g, "\n") !== formatted.replace(/\r\n/g, "\n")) {
    console.error("src/lib/api/schema.ts is out of date: run npm run gen:api");
    process.exit(1);
  }
  console.log("src/lib/api/schema.ts is up to date");
}
