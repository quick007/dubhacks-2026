import type { Persona } from "@storefront/contracts";
import { cn } from "../lib/utils";

const STAGGER_MS = 140;
// Hand-set indents so the searches fall like a crowd of voices, not a list.
const indents = ["0%", "10%", "6%", "30%", "12%", "38%", "3%", "24%", "44%", "9%"];

/** Every shopper's own search, the featured shopper first and loudest. */
export function ShopperQueries({
  personas,
  featuredId,
}: {
  personas: Persona[];
  featuredId: string;
}) {
  const ordered = [
    ...personas.filter((persona) => persona.id === featuredId),
    ...personas.filter((persona) => persona.id !== featuredId),
  ];
  return (
    <ul aria-label="What shoppers search for" className="space-y-1 sm:space-y-2">
      {ordered.map((persona, i) => {
        const lead = i === 0;
        return (
          <li
            key={persona.id}
            style={{
              animationDelay: `${200 + i * STAGGER_MS}ms`,
              paddingLeft: `min(${indents[i % indents.length]}, 30vw)`,
              fontSize: `clamp(1rem, ${2.9 - i * 0.16}vw, ${2.9 - i * 0.16}rem)`,
            }}
            className={cn(
              "flex animate-rise items-baseline gap-3 leading-tight",
              !lead && "max-sm:[&:nth-child(n+6)]:hidden",
            )}
          >
            <span
              className={cn(
                "min-w-0 truncate font-voice italic",
                lead ? "text-ink" : "text-graphite",
              )}
              style={{ opacity: lead ? 1 : 1 - i * 0.045 }}
            >
              “{persona.task?.query ?? persona.goal}”
            </span>
            <span className={cn("shrink-0 text-[0.5em]", lead ? "text-cobalt" : "text-muted")}>
              {persona.name}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
