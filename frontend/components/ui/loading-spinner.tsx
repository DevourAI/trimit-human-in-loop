"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

const spinnerSizes = {
  small: "w-4 h-4 border-2",
  medium: "w-8 h-8 border-4",
  large: "w-16 h-16 border-8",
  xlarge: "w-24 h-24 border-12",
};

const spinnerColors = {
  primary: "border-border border-t-primary",
  secondary: "border-border border-t-secondary",
  success: "border-border border-t-success",
  danger: "border-border border-t-danger",
  warning: "border-border border-t-warning",
  info: "border-border border-t-info",
};

interface LoadingSpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  className?: string;
  size?: keyof typeof spinnerSizes;
  color?: keyof typeof spinnerColors;
}

const LoadingSpinner = React.forwardRef<HTMLDivElement, LoadingSpinnerProps>(
  (props, ref) => {
    const { className, size = "medium", color = "primary", ...rest } = props;
    const spinnerClasses = cn(
      "rounded-full animate-spin",
      spinnerSizes[size],
      spinnerColors[color],
      className
    );
    return <div ref={ref} className={spinnerClasses} {...rest} />;
  }
);

LoadingSpinner.displayName = "LoadingSpinner";

export { LoadingSpinner };
