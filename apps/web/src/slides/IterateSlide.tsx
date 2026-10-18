import { useMemo } from "react";
import { describePatch } from "../lib/format";
import { batchStats, observed, verdict, type Pair, type Session } from "../lib/story";
import { SlideFrame } from "./SlideFrame";

export function IterateSlide({
  pairs,
  featured,
  list,
}: {
  pairs: Pair[];
  featured: Pair;
  list: Session[];
}) {
  const { seen, result, stats } = useMemo(
    () => ({ seen: observed(pairs), result: verdict(pairs), stats: batchStats(list) }),
    [pairs, list],
  );
  const proposal = featured.baseline?.proposedChanges[0];
  const steps = [
    ["Observe", `${seen.gaveUp} of ${seen.shoppers} shoppers gave up on today's shop`],
    [
      "Propose",
      proposal && featured.baseline
        ? describePatch(featured.baseline.config, proposal.configPatch).join(", ")
        : "No change suggested yet",
    ],
    ["Edit", "One setting changed, using facts already in the catalog"],
    [
      "Retest",
      result.total > 0
        ? `${result.total} new shoppers: ${result.before} bought before, ${result.after} after`
        : "Waiting for new shoppers",
    ],
  ];
  return (
    <SlideFrame
      title="Then do it again"
      lead={
        stats.seconds === undefined
          ? `${stats.total} sessions per round, up to ${stats.peak} at once.`
          : `This round: ${stats.total} sessions, up to ${stats.peak} at once, done in ${stats.seconds} s. Round 2 starts from the fixed shop. A live A/B test needs weeks of real traffic for one comparison.`
      }
    >
      <ol className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
        {steps.map(([title, detail], i) => (
          <li key={title} className="border-t-2 border-ink pt-4">
            <p className="font-display text-5xl font-extrabold text-graphite">{i + 1}</p>
            <h3 className="mt-2 text-xl font-bold">{title}</h3>
            <p className="mt-1 text-muted">{detail}</p>
          </li>
        ))}
      </ol>
    </SlideFrame>
  );
}
