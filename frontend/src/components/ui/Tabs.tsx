"use client";

import React, { useId, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/cn";

interface TabsProps<T extends string> {
  value: T;
  onChange: (value: T) => void;
  tabs: { id: T; label: string; count?: number }[];
  className?: string;
  layoutId?: string;
  ariaLabel?: string;
  /** When supplied, tabs receive ids and aria-controls matching `${panelId}-${id}`. */
  panelId?: string;
}

export function Tabs<T extends string>({
  value,
  onChange,
  tabs,
  className,
  layoutId,
  ariaLabel = "Section",
  panelId,
}: TabsProps<T>) {
  const autoId = useId();
  const group = layoutId ?? autoId;
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);

  const focusTab = (index: number) => {
    const next = (index + tabs.length) % tabs.length;
    onChange(tabs[next].id);
    buttons.current[next]?.focus();
  };

  const handleKeyDown = (event: React.KeyboardEvent, index: number) => {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      focusTab(index + 1);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      focusTab(index - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      focusTab(0);
    } else if (event.key === "End") {
      event.preventDefault();
      focusTab(tabs.length - 1);
    }
  };

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn(
        "data-scroll flex items-center gap-1 overflow-x-auto border-b border-edge px-2 sm:px-3",
        className
      )}
    >
      {tabs.map((tab, index) => {
        const active = tab.id === value;
        return (
          <button
            key={tab.id}
            ref={(node) => {
              buttons.current[index] = node;
            }}
            id={panelId ? `tab-${panelId}-${tab.id}` : undefined}
            role="tab"
            type="button"
            aria-selected={active}
            aria-controls={panelId ? `${panelId}-${tab.id}` : undefined}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(tab.id)}
            onKeyDown={(event) => handleKeyDown(event, index)}
            className={cn(
              "relative shrink-0 px-3 py-2.5 text-xs font-semibold transition-colors",
              "focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent",
              active ? "text-accent" : "text-ink-muted hover:text-ink"
            )}
          >
            <span className="flex items-center gap-1.5">
              {tab.label}
              {tab.count !== undefined && (
                <span
                  className={cn(
                    "rounded-full px-1.5 py-px text-[10px] tabular",
                    active ? "bg-accent-soft text-accent" : "bg-inset text-ink-faint"
                  )}
                >
                  {tab.count}
                </span>
              )}
            </span>
            {active && (
              <motion.span
                layoutId={`tab-${group}`}
                className="absolute inset-x-1 -bottom-px h-0.5 rounded-full bg-accent"
                transition={{ type: "spring", stiffness: 420, damping: 34 }}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}
