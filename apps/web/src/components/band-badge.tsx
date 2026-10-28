import type { Band } from "@/lib/api/types";
import { BAND_CLASSES } from "@/lib/bands";
import { cn } from "@/lib/utils";

/** A band label in the band's colour, optionally followed by the priority ("Critical 98"). */
export function BandBadge({
  band,
  priority,
  className,
}: {
  band: Band;
  priority?: number;
  className?: string;
}) {
  return (
    <span
      data-band={band}
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium whitespace-nowrap",
        BAND_CLASSES[band],
        className,
      )}
    >
      {band}
      {priority !== undefined && <span className="tabular-nums">{priority}</span>}
    </span>
  );
}
