import type { Run } from "@storefront/contracts";
import { RotateCcw } from "lucide-react";
import { Button } from "../components/ui/button";
import { batchStats, type Session } from "../lib/story";
import { armLabels } from "./ArmToggle";
import { SessionTile } from "./SessionTile";
import { SlideFrame } from "./SlideFrame";

const STAGGER_MS = 120;

export function SessionsSlide({
  list,
  name,
  personaId,
  busy,
  onEvidence,
  onRun,
}: {
  list: Session[];
  name: string;
  personaId: string;
  busy: boolean;
  onEvidence: (run: Run) => void;
  onRun: () => void;
}) {
  const stats = batchStats(list);
  const shoppers = new Set(list.map((session) => session.persona.id)).size;
  return (
    <SlideFrame
      className="max-w-[1600px] py-5"
      title={`${name} is one of ${shoppers}`}
      actions={
        <Button variant="outline" disabled={busy} onClick={onRun}>
          <RotateCcw />
          {busy ? "Running" : "Run a round"}
        </Button>
      }
    >
      <dl className="flex flex-wrap gap-x-12 gap-y-4">
        {(
          [
            ["Sessions", stats.total],
            ["Running at once", stats.peak],
            ["Seconds, start to finish", stats.seconds ?? "…"],
          ] as const
        ).map(([label, value]) => (
          <div key={label}>
            <dt className="text-muted">{label}</dt>
            <dd className="font-display text-5xl font-extrabold tracking-[-0.015em]">{value}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-6 grid gap-6 lg:grid-cols-2 lg:gap-8">
        {(["baseline", "candidate"] as const).map((arm) => (
          <div key={arm}>
            <h3 className="mb-2 font-bold">{armLabels[arm]}</h3>
            <ul className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
              {list
                .filter((session) => session.arm === arm)
                .map((session, i) => (
                  <li key={session.persona.id}>
                    <SessionTile
                      key={session.run?.id ?? "not-run"}
                      session={session}
                      delay={i * STAGGER_MS + (arm === "candidate" ? STAGGER_MS / 2 : 0)}
                      highlight={session.persona.id === personaId}
                      onOpen={() => session.run && onEvidence(session.run)}
                    />
                    <p className="mt-1 truncate text-sm text-muted">{session.persona.name}</p>
                  </li>
                ))}
            </ul>
          </div>
        ))}
      </div>
    </SlideFrame>
  );
}
