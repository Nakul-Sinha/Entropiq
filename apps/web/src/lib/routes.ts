/** URL builders for every page, and parsing of analysis paths for the navigation. */

const enc = encodeURIComponent;

export const routes = {
  home: () => "/",
  analysis: (id: string) => `/analyses/${enc(id)}`,
  files: (id: string) => `/analyses/${enc(id)}/files`,
  file: (id: string, fileId: number) => `/analyses/${enc(id)}/files/${fileId}`,
  map: (id: string) => `/analyses/${enc(id)}/map`,
  graph: (id: string) => `/analyses/${enc(id)}/graph`,
  plan: (id: string) => `/analyses/${enc(id)}/plan`,
  trend: (repoId: string) => `/repositories/${enc(repoId)}/trend`,
};

export type AnalysisSection = "overview" | "files" | "map" | "graph" | "plan";

export const ANALYSIS_SECTIONS: readonly {
  section: AnalysisSection;
  label: string;
  href: (id: string) => string;
}[] = [
  { section: "overview", label: "Overview", href: routes.analysis },
  { section: "files", label: "Files", href: routes.files },
  { section: "map", label: "Map", href: routes.map },
  { section: "graph", label: "Graph", href: routes.graph },
  { section: "plan", label: "Plan", href: routes.plan },
];

/**
 * The analysis id and section of a pathname, or null outside the analysis pages.
 * "/analyses/x" is the overview; "/analyses/x/files/12" belongs to files. A trailing slash is
 * tolerated and the id is decoded.
 */
export function parseAnalysisPath(
  pathname: string,
): { analysisId: string; section: AnalysisSection } | null {
  const parts = pathname.replace(/\/+$/, "").split("/");
  if (parts[0] !== "" || parts[1] !== "analyses" || !parts[2]) return null;
  let analysisId: string;
  try {
    analysisId = decodeURIComponent(parts[2]);
  } catch {
    return null;
  }
  const rest = parts.slice(3);
  if (rest.length === 0) return { analysisId, section: "overview" };
  if (rest[0] === "files" && rest.length <= 2) return { analysisId, section: "files" };
  if ((rest[0] === "map" || rest[0] === "graph" || rest[0] === "plan") && rest.length === 1) {
    return { analysisId, section: rest[0] };
  }
  return null;
}
