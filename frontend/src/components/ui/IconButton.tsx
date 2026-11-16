import React from "react";
import { cn } from "@/lib/cn";

type Variant = "ghost" | "outline" | "solid";
type Size = "sm" | "md";

const VARIANTS: Record<Variant, string> = {
  ghost:
    "border border-transparent bg-transparent text-ink-muted hover:bg-inset hover:text-ink",
  outline:
    "border border-edge bg-surface text-ink-muted hover:border-edge-strong hover:bg-inset hover:text-ink",
  solid:
    "border border-accent-line bg-accent-soft text-accent hover:border-accent/40 hover:bg-accent-soft/80",
};

const SIZES: Record<Size, string> = {
  sm: "size-8 rounded-lg",
  md: "size-9 rounded-lg",
};

export interface IconButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /** Every icon-only control must carry an accessible name. */
  "aria-label": string;
  variant?: Variant;
  size?: Size;
}

export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton(
    { variant = "outline", size = "md", className, type = "button", ...props },
    ref
  ) {
    return (
      <button
        ref={ref}
        type={type}
        className={cn(
          "inline-grid shrink-0 place-items-center transition-colors duration-150",
          "disabled:pointer-events-none disabled:opacity-40",
          VARIANTS[variant],
          SIZES[size],
          className
        )}
        {...props}
      />
    );
  }
);
