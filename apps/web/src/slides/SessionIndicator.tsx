import type { Session } from "../lib/story";
import { cn } from "../lib/utils";

const dot = {
  bought: "bg-cobalt",
  missed: "bg-signal",
  broke: "bg-muted",
  pending: "bg-graphite animate-pulse",
};

/** Keeps the sample in view: the featured shopper is one of many sessions. Links to the sessions slide. */
export function SessionIndicator({ list, personaId }: { list: Session[]; personaId: string }) {
  const name = list.find((session) => session.persona.id === personaId)?.persona.name;
  const shoppers = new Set(list.map((session) => session.persona.id)).size;
  return (
    <a
      href="#sessions"
      className="hidden items-center gap-3 rounded-md px-2 py-1 hover:bg-paper md:flex"
    >
      <span className="text-muted">
        {name} is 1 of {shoppers} shoppers, {list.length} sessions
      </span>
      <span aria-hidden className="flex gap-0.5">
        {list.map((session) => (
          <span
            key={`${session.persona.id}-${session.arm}`}
            className={cn(
              "size-2 rounded-full",
              session.state ? dot[session.state] : "bg-line",
              session.persona.id === personaId && "ring-2 ring-ink ring-offset-1 ring-offset-fog",
            )}
          />
        ))}
      </span>
    </a>
  );
}
