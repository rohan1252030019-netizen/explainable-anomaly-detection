import React from "react";
import { LucideIcon } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  sub?: string;
  icon?: LucideIcon;
  accentColor?: string;
  trend?: {
    value: string;
    positive?: boolean;
  };
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  sub,
  icon: Icon,
  accentColor,
  trend,
}) => {
  return (
    <div className="saas-card p-5 group relative overflow-hidden">
      {/* Subtle top indicator bar if accent color provided */}
      {accentColor && (
        <div
          className="absolute top-0 left-0 right-0 h-0.5 opacity-80"
          style={{ backgroundColor: accentColor }}
        />
      )}

      <div className="flex items-start justify-between">
        <div>
          <span className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
            {label}
          </span>
          <div
            className="text-2xl font-bold tracking-tight mt-1 text-slate-900 dark:text-slate-50"
            style={accentColor ? { color: accentColor } : undefined}
          >
            {value}
          </div>
        </div>

        {Icon && (
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center bg-slate-100 dark:bg-slate-800/80 text-slate-600 dark:text-slate-300 group-hover:scale-105 transition-transform"
            style={accentColor ? { color: accentColor } : undefined}
          >
            <Icon size={18} />
          </div>
        )}
      </div>

      <div className="mt-3 flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
        {trend && (
          <span
            className={`font-semibold flex items-center ${
              trend.positive ? "text-emerald-500" : "text-rose-500"
            }`}
          >
            {trend.value}
          </span>
        )}
        {sub && <span>{sub}</span>}
      </div>
    </div>
  );
};
