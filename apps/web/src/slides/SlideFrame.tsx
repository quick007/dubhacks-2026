import type { ReactNode } from "react";
import { cn } from "../lib/utils";

/** Shared slide layout: one heading, an optional lead line, actions on the right, then the slide's one idea. */
export function SlideFrame({
  title,
  lead,
  actions,
  children,
  className,
}: {
  title: ReactNode;
  lead?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "mx-auto flex min-h-full w-full max-w-6xl flex-col justify-center px-4 py-10 sm:px-8",
        className,
      )}
    >
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="font-display text-4xl leading-[1.15] font-semibold tracking-[-0.02em] text-balance sm:text-5xl">
            {title}
          </h2>
          {lead && <p className="mt-4 max-w-[60ch] text-lg text-muted sm:text-xl">{lead}</p>}
        </div>
        {actions}
      </div>
      {children && <div className="mt-8">{children}</div>}
    </div>
  );
}
