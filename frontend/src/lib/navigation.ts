import {
  Activity,
  AlertTriangle,
  Boxes,
  Database,
  FileBadge,
  FileText,
  FlaskConical,
  Gauge,
  GitCompare,
  History,
  Layers,
  LayoutDashboard,
  LineChart,
  Network,
  Play,
  Radio,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  UserCheck,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  id: string;
  label: string;
  description: string;
  icon: LucideIcon;
}

export interface NavGroup {
  title: string;
  items: NavItem[];
}

/**
 * Single source of truth for navigation, page headers and the command
 * palette. Adding a screen here is enough for all three surfaces to agree.
 */
export const NAV_GROUPS: NavGroup[] = [
  {
    title: "Overview",
    items: [
      {
        id: "dashboard",
        label: "Dashboard",
        description:
          "Detection results, score distribution and severity mix for the active run.",
        icon: LayoutDashboard,
      },
    ],
  },
  {
    title: "Pipeline",
    items: [
      {
        id: "dataset",
        label: "Dataset",
        description:
          "Upload, profile and preprocess the tabular data used by every detector.",
        icon: Database,
      },
      {
        id: "detection",
        label: "Detection",
        description:
          "Choose an unsupervised detector or intelligent ensemble, configure scoring and run detection.",
        icon: Play,
      },
      {
        id: "anomalies",
        label: "Anomalies",
        description: "Search, filter and rank every flagged record.",
        icon: AlertTriangle,
      },
      {
        id: "investigate",
        label: "Investigate",
        description:
          "Record-level evidence: feature contributions, counterfactuals and explanation checks.",
        icon: Search,
      },
      {
        id: "modelzoo",
        label: "Model Zoo",
        description:
          "Compare 17 production detectors across Classical, Ensemble, and Deep Learning paradigms.",
        icon: Boxes,
      },
    ],
  },
  {
    title: "Intelligence",
    items: [
      {
        id: "explainability",
        label: "Explainability",
        description:
          "Global feature attribution across flagged observations.",
        icon: Sparkles,
      },
      {
        id: "interactions",
        label: "SHAP Interactions",
        description:
          "2nd-order non-linear pairwise interaction values and synergy analysis.",
        icon: GitCompare,
      },
      {
        id: "conformal",
        label: "Conformal & FDR",
        description:
          "Finite-sample valid p-values and Benjamini-Hochberg FDR control.",
        icon: Gauge,
      },
      {
        id: "rootcause",
        label: "Root Causes",
        description:
          "HDBSCAN clustering in SHAP space to discover recurring failure archetypes.",
        icon: Network,
      },
      {
        id: "analytics",
        label: "Visual Analytics",
        description:
          "PCA projection and feature correlation for the active dataset.",
        icon: LineChart,
      },
      {
        id: "models",
        label: "Model Lab",
        description:
          "Benchmark all detectors, compare agreement and inspect overlap.",
        icon: Layers,
      },
      {
        id: "drift",
        label: "Drift Monitor",
        description:
          "Covariate shift and score distribution drift analysis via PSI and 2-sample KS tests.",
        icon: Activity,
      },
      {
        id: "benchmarks",
        label: "Benchmark Lab",
        description:
          "Controlled synthetic anomaly injection laboratory for reproducible detector sensitivity testing.",
        icon: FlaskConical,
      },
      {
        id: "rigor",
        label: "Validation",
        description:
          "Held-out metrics, cross-model comparison and explanation stability.",
        icon: ShieldCheck,
      },
    ],
  },
  {
    title: "Operations",
    items: [
      {
        id: "streaming",
        label: "Streaming Monitor",
        description:
          "Real-time HS-Trees online scoring, sliding PSI drift alarms and recalibration approval.",
        icon: Radio,
      },
      {
        id: "feedback",
        label: "Active Learning",
        description:
          "Analyst triage queue with uncertainty/disagreement sampling and semi-supervised updating.",
        icon: UserCheck,
      },
    ],
  },
  {
    title: "Output",
    items: [
      {
        id: "modelcards",
        label: "Model Cards",
        description:
          "Mitchell et al. model governance cards with dataset fingerprinting and limitation disclosures.",
        icon: FileBadge,
      },
      {
        id: "experiments",
        label: "Experiments",
        description:
          "Reproducible execution registry with cryptographic SHA-256 dataset fingerprints and parameter history.",
        icon: History,
      },
      {
        id: "reports",
        label: "Audit Reports",
        description: "Compile an audit-ready summary of the current analysis.",
        icon: FileText,
      },
      {
        id: "settings",
        label: "Settings",
        description:
          "Theme, exports, experiment history and session controls.",
        icon: Settings,
      },
    ],
  },
];

export const NAV_ITEMS: NavItem[] = NAV_GROUPS.flatMap((group) => group.items);

export function getNavItem(id: string): NavItem | undefined {
  return NAV_ITEMS.find((item) => item.id === id);
}

export function getNavGroup(id: string): string | undefined {
  return NAV_GROUPS.find((group) =>
    group.items.some((item) => item.id === id)
  )?.title;
}
