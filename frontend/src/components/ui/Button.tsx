import React from "react";
import { cn } from "@/lib/cn";

type Variant =
  | "primary"
  | "secondary"
  | "ghost"
  | "outline"
  | "danger"
  | "success";
type Size = "xs" | "sm" | "md" | "lg";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-accent text-on-accent hover:bg-accent-strong border border-transparent shadow-sm",
  secondary:
    "bg-inset text-ink border border-edge hover:border-edge-strong hover:bg-elevated",
  outline:
    "bg-transparent text-ink border border-edge-strong hover:bg-inset",
  ghost: "bg-transparent text-ink-muted border border-transparent hover:bg-inset hover:text-ink",
  danger:
    "bg-critical text-white dark:text-ink-inverse border border-transparent hover:opacity-90 shadow-sm",
  success:
    "bg-good text-white dark:text-ink-inverse border border-transparent hover:opacity-90 shadow-sm",
};

const SIZES: Record<Size, string> = {
  xs: "h-7 px-2.5 text-[11px] gap-1.5 rounded-md",
  sm: "h-8 px-3 text-xs gap-1.5 rounded-lg",
  md: "h-9 px-4 text-xs gap-2 rounded-lg",
  lg: "h-11 px-6 text-sm gap-2 rounded-xl",
};

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    { variant = "primary", size = "md", className, type = "button", ...props },
    ref
  ) {
    return (
      <button
        ref={ref}
        type={type}
        className={cn(
          "inline-flex items-center justify-center font-semibold tracking-tight select-none",
          "transition-[background-color,border-color,color,box-shadow,transform] duration-150 ease-out-expo",
          "active:translate-y-px disabled:pointer-events-none disabled:opacity-45",
          VARIANTS[variant],
          SIZES[size],
          className
        )}
        {...props}
      />
    );
  }
);
