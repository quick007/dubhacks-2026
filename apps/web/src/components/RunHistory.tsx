import type { Run } from "@storefront/contracts";
import { percent, relativeTime } from "../lib/format";
import { outcomeOf } from "../lib/runs";
import { cn } from "../lib/utils";

const dot = { passed: "bg-cobalt", failed: "bg-signal", error: "bg-muted", pending: "bg-graphite" };

export function RunHistory({
  runs,
  selectedId,
  onSelect,
}: {
  runs: Run[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  return (
    <ul className="-mx-2">
      {runs.map((run) => (
        <li key={run.id}>
          <button
            type="button"
            onClick={() => onSelect(run.id)}
            aria-current={run.id === selectedId}
            className={cn(
              "flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-[13px] hover:bg-fog",
              run.id === selectedId && "bg-fog font-medium",
            )}
          >
            <span className={cn("size-2 rounded-full", dot[outcomeOf(run)])} />
            {relativeTime(run.createdAt)}
            <span className="ml-auto text-muted">
              {run.evaluation ? percent(run.evaluation.score) : run.status}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
