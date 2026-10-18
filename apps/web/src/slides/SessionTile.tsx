import { Check, LoaderCircle, TriangleAlert, X } from "lucide-react";
import { useEffect, useState } from "react";
import { frames, type Session } from "../lib/story";
import { useReducedMotion } from "../lib/useReducedMotion";
import { cn } from "../lib/utils";

const FLIP_MS = 650;
const badges = {
  bought: { icon: Check, className: "bg-cobalt text-paper", label: "Bought it" },
  missed: { icon: X, className: "bg-signal text-paper", label: "Did not get it" },
  broke: { icon: TriangleAlert, className: "bg-muted text-paper", label: "Run broke" },
};

/** Replay recorded steps once, then hold the final outcome until the next run. */
export function SessionTile({
  session,
  delay,
  highlight,
  onOpen,
}: {
  session: Session;
  delay: number;
  highlight: boolean;
  onOpen: () => void;
}) {
  const list = frames(session.run);
  const reduced = useReducedMotion();
  const [step, setStep] = useState(0);
  const settled = session.state && session.state !== "pending";
  const shown = reduced || !settled ? list.length - 1 : Math.min(step, list.length - 1);

  useEffect(() => {
    if (reduced || !settled || step >= list.length - 1) return;
    const timer = window.setTimeout(() => setStep(step + 1), FLIP_MS + (step === 0 ? delay : 0));
    return () => window.clearTimeout(timer);
  }, [reduced, settled, step, list.length, delay]);

  const badge =
    session.state && session.state !== "pending" && shown === list.length - 1
      ? badges[session.state]
      : undefined;
  const src = list[shown]?.src;
  return (
    <button
      type="button"
      disabled={!session.run}
      onClick={onOpen}
      aria-label={`${session.persona.name}: ${badge?.label ?? (session.run ? "running" : "not run")}`}
      className={cn(
        "group relative block aspect-[8/5] w-full overflow-hidden rounded border border-line bg-paper",
        highlight && "ring-2 ring-cobalt ring-offset-2 ring-offset-fog",
        !session.run && "border-dashed bg-transparent",
      )}
    >
      {src && <img src={src} alt="" className="size-full object-cover" />}
      {session.state === "pending" && (
        <span className="absolute inset-0 grid place-items-center bg-fog/70">
          <LoaderCircle className="size-5 animate-spin text-muted" />
        </span>
      )}
      {badge && (
        <span
          className={cn(
            "absolute right-1 bottom-1 grid size-5 place-items-center rounded-full",
            badge.className,
          )}
        >
          <badge.icon className="size-3.5" strokeWidth={3} />
        </span>
      )}
    </button>
  );
}
