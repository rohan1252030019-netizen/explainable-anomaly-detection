"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  CornerDownLeft,
  Moon,
  RefreshCw,
  Search,
  Sun,
  Upload,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { api } from "@/lib/api";
import { NAV_GROUPS } from "@/lib/navigation";
import { useToast } from "@/components/ui/Toast";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

interface PaletteAction {
  id: string;
  name: string;
  category: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  run: () => void | Promise<void>;
}

function usePaletteActions(): PaletteAction[] {
  const { setActiveTab, activeTab, theme, toggleTheme, refreshAll } = useApp();
  const { toast } = useToast();

  return useMemo(
    () => [
      ...NAV_GROUPS.flatMap((group) =>
        group.items.map((item) => ({
          id: item.id,
          name: `Go to ${item.label}`,
          category: group.title,
          icon: item.icon,
          run: () => setActiveTab(item.id),
        }))
      ),
      {
        id: "sample",
        name: "Load benchmark sample dataset",
        category: "Actions",
        icon: Upload,
        run: async () => {
          try {
            await api.loadSample();
            await refreshAll();
            setActiveTab("dashboard");
            toast("success", "Sample dataset loaded");
          } catch (err) {
            toast("error", "Could not load the sample", errorMessage(err));
          }
        },
      },
      {
        id: "refresh",
        name: "Refresh data from the API",
        category: "Actions",
        icon: RefreshCw,
        run: async () => {
          await refreshAll();
          toast("info", "Data refreshed");
        },
      },
      {
        id: "theme",
        name: `Switch to ${theme === "dark" ? "light" : "dark"} mode`,
        category: "Interface",
        icon: theme === "dark" ? Sun : Moon,
        run: toggleTheme,
      },
    ].map((action) =>
      action.id === activeTab ? { ...action, name: `${action.name} (current)` } : action
    ),
    [setActiveTab, activeTab, theme, toggleTheme, refreshAll, toast]
  );
}

/**
 * Mounted only while the palette is open, so its local state (query, cursor)
 * resets naturally on close without an effect.
 */
function PalettePanel({ onClose }: { onClose: () => void }) {
  const actions = usePaletteActions();
  const { toast } = useToast();
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const timer = window.setTimeout(() => inputRef.current?.focus(), 20);

    return () => {
      clearTimeout(timer);
      document.body.style.overflow = overflow;
      previous?.focus();
    };
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return actions;
    return actions.filter(
      (action) =>
        action.name.toLowerCase().includes(q) ||
        action.category.toLowerCase().includes(q)
    );
  }, [actions, query]);

  const execute = async (action: PaletteAction) => {
    onClose();
    try {
      await action.run();
    } catch (err) {
      toast("error", "Command failed", errorMessage(err));
    }
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setCursor((current) =>
        filtered.length ? (current + 1) % filtered.length : 0
      );
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setCursor((current) =>
        filtered.length ? (current - 1 + filtered.length) % filtered.length : 0
      );
    } else if (event.key === "Enter" && filtered[cursor]) {
      event.preventDefault();
      void execute(filtered[cursor]);
    } else if (event.key === "Escape") {
      event.preventDefault();
      onClose();
    } else if (event.key === "Tab" && panelRef.current) {
      const focusable = Array.from(
        panelRef.current.querySelectorAll<HTMLElement>(
          'input, button, [href], [tabindex]:not([tabindex="-1"])'
        )
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  };

  return (
    <motion.div
      ref={panelRef}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      initial={{ opacity: 0, scale: 0.985, y: -6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.99, y: -4 }}
      transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
      className="panel w-full max-w-lg overflow-hidden !bg-surface shadow-2xl"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={onKeyDown}
    >
      <div className="flex items-center gap-3 border-b border-edge px-4">
        <Search size={16} className="shrink-0 text-ink-faint" aria-hidden="true" />
        <input
          ref={inputRef}
          type="text"
          role="combobox"
          aria-label="Search commands"
          aria-expanded="true"
          aria-autocomplete="list"
          aria-controls="command-palette-list"
          aria-activedescendant={
            filtered[cursor] ? `cmd-${filtered[cursor].id}` : undefined
          }
          placeholder="Search views and actions…"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setCursor(0);
          }}
          className="w-full bg-transparent py-3.5 text-sm text-ink placeholder:text-ink-faint focus:outline-none"
        />
        <kbd className="hidden shrink-0 rounded border border-edge bg-inset px-1.5 py-0.5 font-mono text-[10px] text-ink-faint sm:block">
          ESC
        </kbd>
      </div>

      <div
        id="command-palette-list"
        role="listbox"
        aria-label="Commands"
        className="data-scroll max-h-[min(24rem,60vh)] overflow-y-auto p-2"
      >
        {filtered.length === 0 ? (
          <p className="py-8 text-center text-xs text-ink-faint">
            No matching commands.
          </p>
        ) : (
          filtered.map((item, index) => {
            const Icon = item.icon;
            const active = index === cursor;
            return (
              <button
                key={item.id}
                id={`cmd-${item.id}`}
                role="option"
                aria-selected={active}
                onMouseEnter={() => setCursor(index)}
                onClick={() => void execute(item)}
                className={cn(
                  "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-xs font-medium transition-colors",
                  active
                    ? "bg-accent-soft text-accent"
                    : "text-ink-muted hover:bg-inset hover:text-ink"
                )}
              >
                <span
                  className={cn(
                    "grid size-7 shrink-0 place-items-center rounded-md border border-edge bg-inset",
                    active ? "text-accent" : "text-ink-faint"
                  )}
                >
                  <Icon size={14} />
                </span>
                <span className="flex-1 truncate">{item.name}</span>
                <span className="shrink-0 text-[9px] uppercase tracking-[0.1em] text-ink-faint">
                  {item.category}
                </span>
              </button>
            );
          })
        )}
      </div>

      <div className="flex items-center justify-between border-t border-edge bg-elevated px-4 py-2 text-[10px] text-ink-faint">
        <span className="flex items-center gap-1.5">
          <kbd className="rounded border border-edge bg-inset px-1 font-mono text-[10px]">
            ↑↓
          </kbd>
          navigate
        </span>
        <span className="flex items-center gap-1 rounded border border-edge bg-inset px-1 font-mono text-[10px]">
          <CornerDownLeft size={9} aria-hidden="true" /> run
        </span>
      </div>
    </motion.div>
  );
}

export const CommandPalette: React.FC = () => {
  const { commandPaletteOpen, setCommandPaletteOpen } = useApp();
  const reducedMotion = useReducedMotion();

  return (
    <AnimatePresence>
      {commandPaletteOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: reducedMotion ? 0 : 0.12 }}
          className="fixed inset-0 z-50 flex items-start justify-center bg-black/55 p-4 pt-[10vh] backdrop-blur-sm sm:pt-[14vh]"
          onClick={() => setCommandPaletteOpen(false)}
        >
          <PalettePanel onClose={() => setCommandPaletteOpen(false)} />
        </motion.div>
      )}
    </AnimatePresence>
  );
};
