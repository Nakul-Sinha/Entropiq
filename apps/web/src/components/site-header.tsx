import Link from "next/link";

import { AnalysisNav } from "@/components/analysis-nav";
import { Badge } from "@/components/ui/badge";
import { USE_MOCKS } from "@/lib/api/client";
import { routes } from "@/lib/routes";
import { MOCK_ANALYSIS_ID } from "@/mocks/ids";

/** The top bar on every page: product name, new analysis, and the analysis sections. */
export function SiteHeader() {
  return (
    <header className="border-b bg-card">
      <div className="mx-auto flex w-full max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href={routes.home()} className="text-lg font-semibold tracking-tight">
          Entropiq
        </Link>
        <Link href={routes.home()} className="text-sm text-muted-foreground hover:text-foreground">
          New analysis
        </Link>
        {USE_MOCKS && (
          <>
            <Link
              href={routes.analysis(MOCK_ANALYSIS_ID)}
              className="text-sm text-muted-foreground hover:text-foreground"
            >
              Demo analysis
            </Link>
            <Badge variant="outline">Mock data</Badge>
          </>
        )}
        <AnalysisNav />
      </div>
    </header>
  );
}
