import type { Metadata, Viewport } from "next";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "SENTINEL — Autonomous Anomaly Intelligence Platform",
  description:
    "SENTINEL — Autonomous Anomaly Intelligence Platform. Detect. Explain. Verify. Act. Unsupervised anomaly detection, conformal validation, and transparent XAI.",
};

export const viewport: Viewport = {
  themeColor: "#f8fafc",
};

const themeScript = `
(function () {
  try {
    var stored = localStorage.getItem("sentinel-theme") || localStorage.getItem("xaud-theme");
    // Light is the product default; dark is an explicit opt-in.
    var dark = stored === "dark";
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
        className="min-h-full bg-background font-sans text-foreground"
        suppressHydrationWarning
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
