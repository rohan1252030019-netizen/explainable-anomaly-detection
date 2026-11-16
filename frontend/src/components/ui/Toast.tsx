"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { cn } from "@/lib/cn";

type ToastTone = "success" | "error" | "info";

interface ToastItem {
  id: number;
  tone: ToastTone;
  title: string;
  description?: string;
}

interface ToastContextValue {
  toast: (tone: ToastTone, title: string, description?: string) => void;
  dismiss: (id: number) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const TONE = {
  success: {
    icon: CheckCircle2,
    accent: "text-good",
    bar: "bg-good",
    role: "status" as const,
  },
  error: {
    icon: AlertTriangle,
    accent: "text-critical",
    bar: "bg-critical",
    role: "alert" as const,
  },
  info: {
    icon: Info,
    accent: "text-accent",
    bar: "bg-accent",
    role: "status" as const,
  },
};

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const nextId = useRef(0);
  const timers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());
  const reducedMotion = useReducedMotion();

  const dismiss = useCallback((id: number) => {
    const timer = timers.current.get(id);
    if (timer) clearTimeout(timer);
    timers.current.delete(id);
    setItems((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const toast = useCallback(
    (tone: ToastTone, title: string, description?: string) => {
      const id = nextId.current++;
      setItems((prev) => [...prev.slice(-2), { id, tone, title, description }]);
      const timer = setTimeout(
        () => dismiss(id),
        tone === "error" ? 8000 : 5000
      );
      timers.current.set(id, timer);
    },
    [dismiss]
  );

  useEffect(() => {
    const pending = timers.current;
    return () => {
      pending.forEach((timer) => clearTimeout(timer));
      pending.clear();
    };
  }, []);

  const value = useMemo(() => ({ toast, dismiss }), [toast, dismiss]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        data-print-hide
        className="pointer-events-none fixed inset-x-0 bottom-0 z-[60] flex flex-col items-center gap-2 p-4 sm:items-end sm:p-6"
        aria-live="polite"
      >
        <AnimatePresence initial={false}>
          {items.map((item) => {
            const tone = TONE[item.tone];
            const Icon = tone.icon;
            return (
              <motion.div
                key={item.id}
                role={tone.role}
                initial={reducedMotion ? { opacity: 0 } : { opacity: 0, y: 10, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={reducedMotion ? { opacity: 0 } : { opacity: 0, y: 6, scale: 0.98 }}
                transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
                className="panel pointer-events-auto relative w-full max-w-sm overflow-hidden p-3.5 pr-10 shadow-xl"
              >
                <span
                  className={cn("absolute inset-y-0 left-0 w-0.5", tone.bar)}
                  aria-hidden="true"
                />
                <div className="flex items-start gap-2.5">
                  <Icon
                    size={15}
                    className={cn("mt-0.5 shrink-0", tone.accent)}
                    aria-hidden="true"
                  />
                  <div className="min-w-0">
                    <p className="text-xs font-bold text-ink">{item.title}</p>
                    {item.description && (
                      <p className="mt-0.5 text-[11px] leading-relaxed text-ink-muted">
                        {item.description}
                      </p>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => dismiss(item.id)}
                  className="absolute right-1.5 top-1.5 grid size-7 place-items-center rounded-md text-ink-faint transition-colors hover:bg-inset hover:text-ink"
                  aria-label="Dismiss notification"
                >
                  <X size={13} />
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) throw new Error("useToast must be used within ToastProvider");
  return context;
}
