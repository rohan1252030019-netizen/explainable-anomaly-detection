import React from "react";

export interface SeverityBadgeProps {
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NORMAL" | string;
  size?: "sm" | "md";
}

export const SeverityBadge: React.FC<SeverityBadgeProps> = ({ severity, size = "md" }) => {
  const sev = (severity || "").toUpperCase();

  let styles = "bg-slate-500/10 text-slate-400 border-slate-500/20";
  let dotColor = "bg-slate-400";

  if (sev === "CRITICAL") {
    styles = "bg-rose-500/15 text-rose-400 border-rose-500/30";
    dotColor = "bg-rose-500";
  } else if (sev === "HIGH") {
    styles = "bg-orange-500/15 text-orange-400 border-orange-500/30";
    dotColor = "bg-orange-500";
  } else if (sev === "MEDIUM") {
    styles = "bg-amber-500/15 text-amber-400 border-amber-500/30";
    dotColor = "bg-amber-500";
  } else if (sev === "LOW" || sev === "NORMAL") {
    styles = "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";
    dotColor = "bg-emerald-500";
  }

  const px = size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-xs";

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-semibold rounded-full border ${px} ${styles}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
      {sev}
    </span>
  );
};
