"use client";

import { Button } from "@/components/ui/button";
import { errorMessage } from "@/lib/api/errors";

/** The default error view of every page. */
export default function ErrorPage({ error, retry }: { error: unknown; retry: () => void }) {
  return (
    <div role="alert" className="mx-auto flex max-w-xl flex-col items-start gap-3 py-12">
      <h1 className="text-2xl font-semibold">Something went wrong</h1>
      <p className="text-muted-foreground">{errorMessage(error)}</p>
      <Button onClick={() => retry()}>Try again</Button>
    </div>
  );
}
