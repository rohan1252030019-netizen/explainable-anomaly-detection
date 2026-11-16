"use client";

import React from "react";
import { MotionConfig } from "framer-motion";
import { AppProvider } from "@/context/AppContext";
import { ToastProvider } from "@/components/ui/Toast";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <MotionConfig reducedMotion="user">
      <ToastProvider>
        <AppProvider>{children}</AppProvider>
      </ToastProvider>
    </MotionConfig>
  );
}
