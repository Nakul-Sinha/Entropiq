import Link from "next/link";

import { routes } from "@/lib/routes";

export default function NotFound() {
  return (
    <div className="mx-auto flex max-w-xl flex-col items-start gap-3 py-12">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <p className="text-muted-foreground">
        This page does not exist, or the analysis has expired.
      </p>
      <Link href={routes.home()} className="text-sm font-medium underline underline-offset-4">
        Start a new analysis
      </Link>
    </div>
  );
}
