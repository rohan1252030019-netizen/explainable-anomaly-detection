import type { Metadata, Viewport } from "next";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "XAUD — Explainable Anomaly Detection Platform",
  description:
    "Production-quality explainable framework for unsupervised anomaly detection with multi-model benchmarking and transparent XAI.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f2f6fa" },
    { media: "(prefers-color-scheme: dark)", color: "#0b1020" },
  ],
};

const themeScript = `
(function () {
  try {
    var stored = localStorage.getItem("xaud-theme");
    var dark = stored ? stored === "dark" : true;
    document.documentElement.classList.toggle("dark", dark);
  } catch (e) {}
})();
`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full antialiased" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body
        className="min-h-full bg-canvas font-sans text-ink"
        suppressHydrationWarning
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
