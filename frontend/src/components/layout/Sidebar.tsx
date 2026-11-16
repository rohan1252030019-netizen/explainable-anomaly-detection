"use client";

import React, { useEffect, useRef } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  PanelLeftClose,
  PanelLeftOpen,
  ShieldCheck,
  X,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { NAV_GROUPS, type NavItem } from "@/lib/navigation";
import { cn } from "@/lib/cn";

const PIPELINE_STEPS = [
  { key: "dataset", label: "Dataset loaded" },
  { key: "features", label: "Features processed" },
  { key: "detection", label: "Detector fitted" },
] as const;

function NavButton({
  item,
  active,
  collapsed,
  badge,
  onSelect,
}: {
  item: NavItem;
  active: boolean;
  collapsed: boolean;
  badge?: number;
  onSelect: () => void;
}) {
  const Icon = item.icon;

  return (
    <button
      onClick={onSelect}
      title={collapsed ? item.label : item.description}
      aria-current={active ? "page" : undefined}
      className={cn(
        "group relative flex w-full items-center rounded-lg text-xs font-semibold transition-colors duration-150",
        collapsed ? "justify-center px-0 py-2.5" : "gap-3 px-3 py-2",
        active
          ? "bg-accent-soft text-accent"
          : "text-ink-muted hover:bg-inset hover:text-ink"
      )}
    >
      {active && !collapsed && (
        <span
          className="absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-accent"
          aria-hidden="true"
        />
      )}
      <Icon
        size={16}
        className={cn(
          "shrink-0 transition-colors",
          active ? "text-accent" : "text-ink-faint group-hover:text-ink-muted"
        )}
        aria-hidden="true"
      />
      {!collapsed && (
        <>
          <span className="flex-1 truncate text-left">{item.label}</span>
          {badge !== undefined && badge > 0 && (
            <span
              className={cn(
                "rounded-full px-1.5 py-px text-[10px] font-bold tabular",
                active
                  ? "bg-accent/15 text-accent"
                  : "border border-critical/25 bg-critical-soft text-critical"
              )}
            >
              {badge > 9999 ? "9k+" : badge}
            </span>
          )}
        </>
      )}
      {collapsed && badge !== undefined && badge > 0 && (
        <span
          className="absolute right-1.5 top-1.5 size-1.5 rounded-full bg-critical"
          aria-hidden="true"
        />
      )}
    </button>
  );
}

function SidebarContent({
  collapsed,
  onToggleCollapse,
  onNavigate,
  onClose,
  closeRef,
}: {
  collapsed: boolean;
  onToggleCollapse: () => void;
  onNavigate?: () => void;
  onClose?: () => void;
  closeRef?: React.RefObject<HTMLButtonElement | null>;
}) {
  const { activeTab, setActiveTab, dataset, overview } = useApp();

  const steps = [
    { ...PIPELINE_STEPS[0], done: !!dataset?.loaded },
    { ...PIPELINE_STEPS[1], done: !!dataset?.feature_matrix_ready },
    { ...PIPELINE_STEPS[2], done: !!overview?.ready },
  ];
  const completed = steps.filter((step) => step.done).length;

  return (
    <>
      {/* Brand */}
      <div
        className={cn(
          "flex items-center border-b border-edge",
          collapsed ? "justify-center px-3 py-[1.05rem]" : "gap-3 px-4 py-4"
        )}
      >
        <span
          className="relative grid size-8 shrink-0 place-items-center rounded-lg bg-accent text-[13px] font-black text-on-accent"
          aria-hidden="true"
        >
          X
          <span className="absolute -right-0.5 -top-0.5 size-1.5 rounded-full border border-sidebar bg-good" />
        </span>
        {!collapsed && (
          <div className="min-w-0">
            <div className="flex items-baseline gap-1.5">
              <span className="text-sm font-extrabold tracking-tight text-ink">
                XAUD
              </span>
              <span className="text-[9px] font-bold uppercase tracking-[0.12em] text-ink-faint">
                2.0
              </span>
            </div>
            <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-ink-faint">
              Anomaly Intelligence
            </div>
          </div>
        )}
        {!collapsed && onClose && (
          <button
            ref={closeRef}
            onClick={onClose}
            className="ml-auto grid size-8 place-items-center rounded-lg text-ink-faint hover:bg-inset hover:text-ink"
            aria-label="Close navigation"
          >
            <X size={16} />
          </button>
        )}
      </div>

      {/* Navigation */}
      <nav
        aria-label="Primary"
        className={cn(
          "data-scroll flex-1 overflow-y-auto py-3",
          collapsed ? "px-2" : "px-2.5"
        )}
      >
        {NAV_GROUPS.map((group, index) => (
          <div
            key={group.title}
            className={cn(index > 0 && "mt-4 border-t border-edge pt-4")}
          >
            {!collapsed ? (
              <div className="mb-1.5 px-3 text-[9px] font-bold uppercase tracking-[0.16em] text-ink-faint">
                {group.title}
              </div>
            ) : (
              <div className="sr-only">{group.title}</div>
            )}
            <ul className="space-y-0.5">
              {group.items.map((item) => (
                <li key={item.id}>
                  <NavButton
                    item={item}
                    active={activeTab === item.id}
                    collapsed={collapsed}
                    badge={
                      item.id === "anomalies" ? overview?.n_anomalies : undefined
                    }
                    onSelect={() => {
                      setActiveTab(item.id);
                      onNavigate?.();
                    }}
                  />
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      {/* Pipeline summary */}
      <div className={cn("border-t border-edge", collapsed ? "p-2" : "p-3")}>
        {collapsed ? (
          <div
            className="flex flex-col items-center gap-1.5 py-1"
            title={`Pipeline ${completed}/${steps.length} complete`}
          >
            {steps.map((step) => (
              <span
                key={step.key}
                className={cn(
                  "size-1.5 rounded-full",
                  step.done ? "bg-good" : "bg-edge-strong"
                )}
              />
            ))}
            <span className="sr-only">
              Pipeline {completed} of {steps.length} steps complete
            </span>
          </div>
        ) : (
          <>
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="eyebrow">Pipeline</span>
              <span className="flex items-center gap-1.5 text-[10px] font-semibold text-ink-muted">
                <ShieldCheck
                  size={11}
                  className={completed === 3 ? "text-good" : "text-ink-faint"}
                  aria-hidden="true"
                />
                {completed}/{steps.length}
              </span>
            </div>

            <div
              className="mb-3 grid grid-cols-3 gap-1"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={3}
              aria-valuenow={completed}
              aria-label="Analysis pipeline progress"
            >
              {steps.map((step) => (
                <span
                  key={step.key}
                  className={cn(
                    "h-1 rounded-full transition-colors",
                    step.done ? "bg-good" : "bg-edge"
                  )}
                />
              ))}
            </div>

            <ol className="space-y-1.5">
              {steps.map((step) => (
                <li
                  key={step.key}
                  className="flex items-center gap-2 text-[10.5px] text-ink-muted"
                >
                  <span
                    className={cn(
                      "size-1 shrink-0 rounded-full",
                      step.done ? "bg-good" : "bg-edge-strong"
                    )}
                    aria-hidden="true"
                  />
                  <span className={cn("truncate", !step.done && "text-ink-faint")}>
                    {step.label}
                  </span>
                  <span className="sr-only">
                    {step.done ? "completed" : "not started"}
                  </span>
                </li>
              ))}
            </ol>

            {dataset?.dataset_name && (
              <div className="mt-3 truncate border-t border-edge pt-2 text-[10px] text-ink-faint">
                <span className="sr-only">Active dataset: </span>
                {dataset.dataset_name}
              </div>
            )}
          </>
        )}

        <button
          onClick={onToggleCollapse}
          className={cn(
            "mt-2 flex w-full items-center rounded-lg py-1.5 text-[10.5px] font-semibold text-ink-faint transition-colors hover:bg-inset hover:text-ink",
            collapsed ? "justify-center" : "gap-2 px-2"
          )}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          title="Toggle sidebar (Ctrl/⌘ + Shift + B)"
        >
          {collapsed ? (
            <PanelLeftOpen size={14} />
          ) : (
            <>
              <PanelLeftClose size={14} />
              <span>Collapse</span>
              <kbd className="ml-auto rounded border border-edge bg-inset px-1 font-mono text-[9px] font-normal">
                ⇧⌘B
              </kbd>
            </>
          )}
        </button>
      </div>
    </>
  );
}

export const Sidebar: React.FC = () => {
  const { sidebarCollapsed, toggleSidebar, mobileNavOpen, setMobileNavOpen } =
    useApp();
  const reducedMotion = useReducedMotion();
  const closeRef = useRef<HTMLButtonElement>(null);
  const drawerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!mobileNavOpen) return;

    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const timer = window.setTimeout(() => closeRef.current?.focus(), 30);

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMobileNavOpen(false);
        return;
      }
      if (event.key !== "Tab" || !drawerRef.current) return;

      const focusable = Array.from(
        drawerRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
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
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      clearTimeout(timer);
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = overflow;
      previous?.focus();
    };
  }, [mobileNavOpen, setMobileNavOpen]);

  return (
    <>
      {/* Desktop */}
      <aside
        className={cn(
          "sticky top-0 hidden h-svh shrink-0 flex-col border-r border-edge bg-sidebar transition-[width] duration-200 ease-out-expo lg:flex",
          sidebarCollapsed ? "w-[68px]" : "w-[264px]"
        )}
        data-print-hide
      >
        <SidebarContent
          collapsed={sidebarCollapsed}
          onToggleCollapse={toggleSidebar}
        />
      </aside>

      {/* Mobile drawer */}
      <AnimatePresence>
        {mobileNavOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: reducedMotion ? 0 : 0.15 }}
              onClick={() => setMobileNavOpen(false)}
              className="fixed inset-0 z-40 bg-black/55 backdrop-blur-sm lg:hidden"
              aria-hidden="true"
            />
            <motion.aside
              ref={drawerRef}
              key="drawer"
              role="dialog"
              aria-modal="true"
              aria-label="Navigation"
              initial={reducedMotion ? { opacity: 0 } : { x: "-100%" }}
              animate={reducedMotion ? { opacity: 1 } : { x: 0 }}
              exit={reducedMotion ? { opacity: 0 } : { x: "-100%" }}
              transition={{ duration: reducedMotion ? 0 : 0.2, ease: [0.16, 1, 0.3, 1] }}
              className="fixed inset-y-0 left-0 z-40 flex w-[280px] flex-col border-r border-edge bg-sidebar shadow-2xl lg:hidden"
            >
              <SidebarContent
                collapsed={false}
                onToggleCollapse={toggleSidebar}
                onNavigate={() => setMobileNavOpen(false)}
                onClose={() => setMobileNavOpen(false)}
                closeRef={closeRef}
              />
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
};
