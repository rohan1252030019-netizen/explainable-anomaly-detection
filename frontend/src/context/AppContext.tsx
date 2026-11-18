"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import {
  api,
  isAbortError,
  type DatasetSummary,
  type DetectionOverview,
  type HealthResponse,
} from "@/lib/api";

type Theme = "dark" | "light";

interface AppContextValue {
  dataset: DatasetSummary | null;
  overview: DetectionOverview | null;
  health: HealthResponse | null;
  theme: Theme;
  toggleTheme: () => void;
  isLoading: boolean;
  activeTab: string;
  setActiveTab: (tab: string) => void;
  commandPaletteOpen: boolean;
  setCommandPaletteOpen: (open: boolean) => void;
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  mobileNavOpen: boolean;
  setMobileNavOpen: (open: boolean) => void;
  refreshDataset: () => Promise<void>;
  refreshOverview: () => Promise<void>;
  refreshAll: () => Promise<void>;
}

const AppContext = createContext<AppContextValue | null>(null);

/* ------------------------------------------------------------------ *
 * Preference stores. localStorage is an external system, so it is read
 * through useSyncExternalStore instead of mirrored into React state.
 * ------------------------------------------------------------------ */

function createBooleanStore(key: string, fallback: boolean) {
  const event = `${key}-change`;

  const subscribe = (onChange: () => void) => {
    window.addEventListener("storage", onChange);
    window.addEventListener(event, onChange);
    return () => {
      window.removeEventListener("storage", onChange);
      window.removeEventListener(event, onChange);
    };
  };

  const getSnapshot = () => {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    return raw === "1" || raw === "true";
  };

  const set = (value: boolean) => {
    localStorage.setItem(key, value ? "1" : "0");
    window.dispatchEvent(new Event(event));
  };

  return { subscribe, getSnapshot, set, event };
}

const themeStore = (() => {
  const key = "sentinel-theme";
  const legacyKey = "xaud-theme";
  const event = `${key}-change`;
  // Light is the product default; dark is an explicit opt-in.
  const fallback: Theme = "light";

  return {
    subscribe: (onChange: () => void) => {
      window.addEventListener("storage", onChange);
      window.addEventListener(event, onChange);
      return () => {
        window.removeEventListener("storage", onChange);
        window.removeEventListener(event, onChange);
      };
    },
    getSnapshot: (): Theme => {
      const raw = localStorage.getItem(key) || localStorage.getItem(legacyKey);
      return raw === "light" || raw === "dark" ? raw : fallback;
    },
    set: (value: Theme) => {
      localStorage.setItem(key, value);
      window.dispatchEvent(new Event(event));
    },
  };
})();

const sidebarStore = createBooleanStore("sentinel-sidebar-collapsed", false);

function applyThemeClass(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [overview, setOverview] = useState<DetectionOverview | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTabState] = useState("dashboard");
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const theme = useSyncExternalStore(
    themeStore.subscribe,
    themeStore.getSnapshot,
    () => "light" as Theme
  );
  const sidebarCollapsed = useSyncExternalStore(
    sidebarStore.subscribe,
    sidebarStore.getSnapshot,
    () => false
  );

  // Keep the root class in sync with the stored preference.
  useEffect(() => {
    applyThemeClass(theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    const next: Theme =
      themeStore.getSnapshot() === "dark" ? "light" : "dark";
    themeStore.set(next);
    applyThemeClass(next);
  }, []);

  const toggleSidebar = useCallback(() => {
    sidebarStore.set(!sidebarStore.getSnapshot());
  }, []);

  const setActiveTab = useCallback((tab: string) => {
    setActiveTabState(tab);
    setMobileNavOpen(false);
  }, []);

  const refreshDataset = useCallback(async (signal?: AbortSignal) => {
    try {
      setDataset(await api.getDatasetSummary({ signal }));
    } catch (e) {
      // An aborted request is not a failure: the component unmounted or the
      // user moved on, and reporting it would be noise.
      if (!isAbortError(e)) console.warn("Could not load dataset summary", e);
    }
  }, []);

  const refreshOverview = useCallback(async (signal?: AbortSignal) => {
    try {
      setOverview(await api.getResults({ signal }));
    } catch (e) {
      if (!isAbortError(e)) console.warn("Could not load detection results", e);
    }
  }, []);

  const loadAll = useCallback(async (signal?: AbortSignal) => {
    const [healthResult, datasetResult, overviewResult] = await Promise.allSettled([
      api.getHealth({ signal }),
      api.getDatasetSummary({ signal }),
      api.getResults({ signal }),
    ]);

    if (signal?.aborted) return;
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    if (datasetResult.status === "fulfilled") setDataset(datasetResult.value);
    if (overviewResult.status === "fulfilled") setOverview(overviewResult.value);
  }, []);

  const refreshAll = useCallback(async () => {
    setIsLoading(true);
    try {
      await loadAll();
    } finally {
      setIsLoading(false);
    }
  }, [loadAll]);

  useEffect(() => {
    // The controller is threaded all the way into `fetch`, so navigating away
    // cancels the in-flight requests instead of letting them resolve into
    // state on an unmounted tree.
    const controller = new AbortController();
    const run = async () => {
      await loadAll(controller.signal);
      if (!controller.signal.aborted) setIsLoading(false);
    };
    void run();
    return () => controller.abort();
  }, [loadAll]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;

      if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCommandPaletteOpen((prev) => !prev);
        return;
      }
      if (mod && e.shiftKey && e.key.toLowerCase() === "b") {
        e.preventDefault();
        toggleSidebar();
        return;
      }
      if (e.key === "Escape") {
        setCommandPaletteOpen(false);
        setMobileNavOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [toggleSidebar]);

  return (
    <AppContext.Provider
      value={{
        dataset,
        overview,
        health,
        theme,
        toggleTheme,
        isLoading,
        activeTab,
        setActiveTab,
        commandPaletteOpen,
        setCommandPaletteOpen,
        sidebarCollapsed,
        toggleSidebar,
        mobileNavOpen,
        setMobileNavOpen,
        refreshDataset,
        refreshOverview,
        refreshAll,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useApp must be used within an AppProvider");
  }
  return context;
}
