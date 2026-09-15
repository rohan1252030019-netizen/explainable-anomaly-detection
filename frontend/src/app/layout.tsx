import type { Metadata } from "next";
import { AppProvider } from "@/context/AppContext";
import "./globals.css";

export const metadata: Metadata = {
  title: "XAUD — Explainable Anomaly Detection Platform",
  description:
    "Production-quality explainable framework for unsupervised anomaly detection with multi-model benchmarking and transparent XAI.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full antialiased" suppressHydrationWarning>
      <body className="min-h-full flex flex-col bg-slate-50 dark:bg-[#090D16] text-slate-900 dark:text-slate-100">
        <AppProvider>{children}</AppProvider>
      </body>
    </html>
  );
}
