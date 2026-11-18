import React from "react";
import { cn } from "@/lib/cn";

export function Card({
  className,
  interactive = false,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & { interactive?: boolean }) {
  return (
    <div
      className={cn(
        "panel",
        interactive &&
          "transition-all duration-200 ease-out-expo hover:border-primary hover:-translate-y-px",
        className
      )}
      {...props}
    />
  );
}

export function CardHeader({
  title,
  description,
  icon: Icon,
  action,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  icon?: React.ComponentType<{ size?: number; className?: string }>;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-start justify-between gap-4 border-b border-border px-5 py-4",
        className
      )}
    >
      <div className="min-w-0">
        <h2 className="flex items-center gap-2 text-sm font-bold tracking-tight text-foreground">
          {Icon && <Icon size={15} className="shrink-0 text-primary" />}
          <span className="truncate">{title}</span>
        </h2>
        {description && (
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            {description}
          </p>
        )}
      </div>
      {action && <div className="flex shrink-0 items-center gap-2">{action}</div>}
    </div>
  );
}

export function SectionLabel({
  children,
  className,
  style,
}: {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div
      style={style}
      className={cn(
        "text-[10px] font-bold uppercase tracking-[0.14em] text-subtle-foreground",
        className
      )}
    >
      {children}
    </div>
  );
}
