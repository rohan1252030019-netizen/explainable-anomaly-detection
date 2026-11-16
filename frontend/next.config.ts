import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The enclosing directory is itself a Git repository, so pin the workspace
  // root to this app to keep Turbopack from walking up to it.
  turbopack: {
    root: path.resolve("."),
  },
};

export default nextConfig;
