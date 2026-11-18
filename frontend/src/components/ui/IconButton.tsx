import React from "react";
import { cn } from "@/lib/cn";

type Variant = "ghost" | "outline" | "solid";
type Size = "sm" | "md";

const VARIANTS: Record<Variant, string> = {
  ghost:
    "border border-transparent bg-transparent text-muted-foreground hover:bg-surface-muted hover:text-foreground",
  outline:
    "border border-border bg-surface text-muted-foreground hover:border-border-strong hover:bg-surface-muted hover:text-foreground",
  solid:
    "border border-primary bg-primary-soft text-primary hover:border-primary hover:bg-surface",
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
