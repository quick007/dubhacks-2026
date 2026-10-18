import type { Persona } from "@storefront/contracts";
import { useEffect, useState } from "react";
import { useReducedMotion } from "../lib/useReducedMotion";

const TYPE_MS = 45;
const TYPE_DELAY_MS = 900;

export function MeetSlide({ persona }: { persona: Persona }) {
  const query = persona.task?.query ?? persona.goal;
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
      <h2 className="font-display text-6xl leading-[0.92] font-extrabold tracking-[-0.02em] sm:text-8xl">
        Meet {persona.name}
      </h2>
      {persona.bio && (
        <p className="mt-6 max-w-[40ch] font-voice text-2xl leading-snug sm:text-3xl">
          {persona.bio}
        </p>
      )}
      <p
        aria-label={query}
        className="mt-14 max-w-[30ch] border-b-2 border-ink pb-3 font-voice text-4xl leading-tight text-cobalt italic sm:text-6xl"
      >
        <span aria-hidden>{reduced ? query : query.slice(0, typed)}</span>
        <span
          aria-hidden
          className="ml-1 inline-block h-[0.8em] w-[0.08em] animate-pulse bg-cobalt align-baseline"
        />
      </p>
    </div>
  );
}
