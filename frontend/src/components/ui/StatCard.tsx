import React from "react";
import { motion } from "framer-motion";
import { TrendingDown, TrendingUp, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/cn";

interface StatCardProps {
  label: string;
  value: string | number;
  sub?: string;
  icon?: LucideIcon;
  /** Optional semantic accent for the icon and top rule only. */
  accentColor?: string;
  trend?: { value: string; positive?: boolean };
  className?: string;
  compact?: boolean;
  /** Allow long text values (for example model names) to wrap instead of truncating. */
  multiline?: boolean;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  sub,
  icon: Icon,
  accentColor,
  trend,
  className,
  compact = false,
  multiline = false,
}) => {
  return (
    <div
      className={cn(
        "panel group relative overflow-hidden transition-colors duration-200 hover:border-border-strong",
        compact ? "p-3.5" : "p-4",
        className
      )}
    >
      {accentColor && (
        <span
          className="absolute inset-x-0 top-0 h-0.5 opacity-80"
          style={{ backgroundColor: accentColor }}
          aria-hidden="true"
        />
      )}

      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <span className="text-[10px] font-semibold uppercase tracking-[0.1em] text-subtle-foreground">
            {label}
          </span>
          <div
            className={cn(
              "mt-1.5 font-bold tracking-tight text-foreground tabular",
              compact || multiline ? "text-lg leading-6" : "text-[1.4rem] leading-7",
              multiline ? "whitespace-normal break-words" : "truncate"
            )}
            title={String(value)}
          >
            {value}
          </div>
        </div>

        {Icon && (
          <span
            className="grid size-8 shrink-0 place-items-center rounded-lg border border-border bg-surface-muted text-subtle-foreground transition-colors group-hover:text-muted-foreground"
            style={accentColor ? { color: accentColor } : undefined}
          >
            <Icon size={15} aria-hidden="true" />
          </span>
        )}
      </div>

      {(trend || sub) && (
        <div className="mt-2.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] leading-snug text-muted-foreground">
          {trend && (
            <span
              className={cn(
                "inline-flex items-center gap-1 font-bold tabular",
                trend.positive ? "text-success" : "text-danger"
              )}
            >
              {trend.positive ? (
                <TrendingUp size={11} aria-hidden="true" />
              ) : (
                <TrendingDown size={11} aria-hidden="true" />
              )}
              {trend.value}
            </span>
          )}
          {sub && <span className="truncate">{sub}</span>}
        </div>
      )}
    </div>
  );
};

export function StatStrip({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        "grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6",
        className
      )}
    >
      {children}
    </motion.div>
  );
}
