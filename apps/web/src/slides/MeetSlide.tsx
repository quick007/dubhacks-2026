import { HugeiconsIcon } from "@hugeicons/react";
import { Search01Icon } from "@hugeicons/core-free-icons";
import type { Persona } from "@storefront/contracts";
import { useEffect, useState } from "react";
import { useReducedMotion } from "../lib/useReducedMotion";

const TYPE_MS = 45;
const TYPE_DELAY_MS = 900;

export function MeetSlide({ persona }: { persona: Persona }) {
  const query = (persona.task?.query ?? persona.goal).replace(/\.$/, "");
  const reduced = useReducedMotion();
  const [typed, setTyped] = useState(0);
  useEffect(() => {
    if (reduced || typed >= query.length) return;
    const timer = window.setTimeout(
      () => setTyped(typed + 1),
      typed === 0 ? TYPE_DELAY_MS : TYPE_MS,
    );
    return () => window.clearTimeout(timer);
  }, [typed, query, reduced]);
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col justify-center px-4 sm:px-8">
      <h2 className="font-display text-5xl leading-[1.15] font-semibold tracking-[-0.02em] text-balance sm:text-7xl">
        Meet {persona.name}
      </h2>
      {persona.bio && (
        <p className="mt-6 max-w-[40ch] font-sans text-2xl leading-snug sm:text-3xl">
          {persona.bio}
        </p>
      )}
      <p
        aria-label={query}
        className="mt-14 max-w-[30ch] border-b-2 border-ink pb-3 font-sans text-4xl leading-tight text-cobalt sm:text-6xl"
      >
        {!reduced && typed === 0 && (
          <HugeiconsIcon
            icon={Search01Icon}
            aria-hidden="true"
            className="mr-3 inline-block size-[0.8em] align-baseline text-muted"
            strokeWidth={1.5}
          />
        )}
        <span aria-hidden>{reduced ? query : query.slice(0, typed)}</span>
        <span
          aria-hidden
          className="ml-1 inline-block h-[0.8em] w-[0.08em] animate-pulse bg-cobalt align-baseline"
        />
      </p>
    </div>
  );
}
