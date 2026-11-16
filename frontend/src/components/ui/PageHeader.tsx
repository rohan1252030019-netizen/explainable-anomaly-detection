import React from "react";
import { cn } from "@/lib/cn";

export type PageStatusTone = "ready" | "active" | "idle" | "warning";

const STATUS_TONES: Record<PageStatusTone, { dot: string; chip: string }> = {
  ready: { dot: "bg-good", chip: "border-good/25 bg-good-soft text-good" },
  active: { dot: "bg-accent", chip: "border-accent-line bg-accent-soft text-accent" },
  idle: { dot: "bg-ink-faint", chip: "border-edge bg-inset text-ink-muted" },
  warning: {
    dot: "bg-caution",
    chip: "border-caution/25 bg-caution-soft text-caution",
  },
};

interface PageHeaderProps {
  group?: string;
  title: React.ReactNode;
  description?: React.ReactNode;
  status?: { tone: PageStatusTone; label: string };
  actions?: React.ReactNode;
  className?: string;
}

/**
 * Shared page identity block. The shell owns it so every screen starts with
 * the same title rhythm and context, while page-specific actions stay in the
 * screen that owns them.
 */
export function PageHeader({
  group,
  title,
  description,
  status,
  actions,
  className,
}: PageHeaderProps) {
  return (
    <header
      data-print-hide
      className={cn(
        "mb-6 flex flex-col gap-4 border-b border-edge pb-5 sm:flex-row sm:items-end sm:justify-between",
        className
      )}
    >
      <div className="min-w-0">
        {group && <div className="eyebrow">{group}</div>}
        <h1 className="mt-1 text-xl font-bold tracking-tight text-ink sm:text-[1.375rem]">
          {title}
        </h1>
        {description && (
          <p className="mt-1.5 max-w-3xl text-xs leading-relaxed text-ink-muted">
            {description}
          </p>
        )}
      </div>

      {(status || actions) && (
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          {status && (
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold",
                STATUS_TONES[status.tone].chip
              )}
            >
              <span
                className={cn("size-1.5 rounded-full", STATUS_TONES[status.tone].dot)}
                aria-hidden="true"
              />
              {status.label}
            </span>
          )}
          {actions}
        </div>
      )}
    </header>
  );
}
