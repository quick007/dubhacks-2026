import type { Persona, Run } from "@storefront/contracts";
import { SlideFrame } from "./SlideFrame";

const MAX_QUOTES = 3;

export function OthersSlide({
  name,
  quotes,
  onEvidence,
}: {
  name: string;
  quotes: { persona: Persona; text: string; run: Run }[];
  onEvidence: (run: Run) => void;
}) {
  return (
    <SlideFrame title={`Not just ${name}`}>
      <ul className="grid gap-x-12 gap-y-10 md:grid-cols-3">
        {quotes.slice(0, MAX_QUOTES).map(({ persona, text, run }) => (
          <li key={persona.id}>
            <blockquote className="font-display text-2xl leading-snug font-bold sm:text-3xl">
              “{text}”
            </blockquote>
            <p className="mt-4 text-muted">
              {persona.name}, looking for {persona.task?.query}.{" "}
              <button
                type="button"
                className="text-ink underline underline-offset-4"
                onClick={() => onEvidence(run)}
              >
                See their steps
              </button>
            </p>
          </li>
        ))}
      </ul>
    </SlideFrame>
  );
}
