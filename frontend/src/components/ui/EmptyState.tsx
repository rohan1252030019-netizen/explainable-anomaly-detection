import React from "react";
import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { Button } from "./Button";
import { cn } from "@/lib/cn";

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
  actionText?: string;
  onAction?: () => void;
  secondaryActionText?: string;
  onSecondaryAction?: () => void;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon: Icon,
  title,
  description,
  actionText,
  onAction,
  secondaryActionText,
  onSecondaryAction,
  className,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        "panel-flush flex flex-col items-center justify-center px-6 py-10 text-center",
        className
      )}
    >
      <div className="mb-4 grid size-11 place-items-center rounded-xl border border-border bg-surface-muted text-subtle-foreground">
        <Icon size={19} aria-hidden="true" />
      </div>

      <h3 className="text-sm font-bold tracking-tight text-foreground">{title}</h3>
      <p className="mt-1.5 max-w-md text-xs leading-relaxed text-muted-foreground">
        {description}
      </p>

      {(actionText && onAction) || (secondaryActionText && onSecondaryAction) ? (
        <div className="mt-5 flex flex-wrap items-center justify-center gap-2.5">
          {actionText && onAction && (
            <Button onClick={onAction} size="sm">
              {actionText}
            </Button>
          )}
          {secondaryActionText && onSecondaryAction && (
            <Button variant="outline" size="sm" onClick={onSecondaryAction}>
              {secondaryActionText}
            </Button>
          )}
        </div>
      ) : null}
    </motion.div>
  );
};
