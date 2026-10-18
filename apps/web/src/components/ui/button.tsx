import * as React from "react";
import { Button as BaseButton } from "@base-ui/react/button";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-2 rounded-md text-sm font-medium whitespace-nowrap transition-colors disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4",
  {
    variants: {
      variant: {
        default: "bg-ink text-paper hover:bg-cobalt",
        outline: "border border-line bg-paper text-ink hover:bg-fog",
        ghost: "text-muted hover:bg-fog hover:text-ink",
      },
      size: { default: "h-9 px-3.5", sm: "h-7 px-2.5 text-xs [&_svg]:size-3.5" },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);
export interface ButtonProps
  extends React.ComponentProps<typeof BaseButton>, VariantProps<typeof buttonVariants> {
  className?: string;
}
export function Button({ className, variant, size, ...props }: ButtonProps) {
  return <BaseButton className={cn(buttonVariants({ variant, size, className }))} {...props} />;
}
