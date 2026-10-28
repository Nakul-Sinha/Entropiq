"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { analysisQuery } from "@/lib/api/queries";
import { ANALYSIS_SECTIONS, parseAnalysisPath, routes } from "@/lib/routes";
import { cn } from "@/lib/utils";

const linkClass = "rounded-md px-2 py-1 text-sm hover:bg-accent hover:text-accent-foreground";

/** Links to the sections of the analysis in the URL, plus its repository trend; nothing elsewhere. */
export function AnalysisNav() {
  const pathname = usePathname();
  const current = parseAnalysisPath(pathname ?? "");
  const analysisId = current?.analysisId ?? null;
  const { data } = useQuery({ ...analysisQuery(analysisId ?? ""), enabled: analysisId !== null });
  if (current === null || analysisId === null) return null;

  return (
    <nav aria-label="Analysis sections" className="flex flex-wrap items-center gap-1 sm:ml-auto">
      {ANALYSIS_SECTIONS.map(({ section, label, href }) => {
        const active = section === current.section;
        return (
          <Link
            key={section}
            href={href(analysisId)}
            aria-current={active ? "page" : undefined}
            className={cn(linkClass, active ? "bg-accent font-medium" : "text-muted-foreground")}
          >
            {label}
          </Link>
        );
      })}
      {data && (
        <Link
          href={routes.trend(data.repository_id)}
          className={cn(linkClass, "text-muted-foreground")}
        >
          Trend
        </Link>
      )}
    </nav>
  );
}
