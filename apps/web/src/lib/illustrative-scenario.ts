import assumptions from "../fieldwork/illustrative-scenario.json";

const round = (n: number) => Math.round(n * 100) / 100;

/** A displayed cross-experiment scenario, never an estimate of measured net revenue. */
export function illustrativeScenario(reportId: string, sha256: string, beforeCents: number, afterCents: number) {
  if (reportId !== assumptions.reportId || sha256 !== assumptions.reportSha256
    || !Number.isFinite(beforeCents) || !Number.isFinite(afterCents) || beforeCents <= 0 || afterCents < 0) return null;
  const observedPercent = round((afterCents / beforeCents - 1) * 100);
  const assumedLossPercent = round(assumptions.displayIncrementalMs / 100 * assumptions.assumedRevenueLossPercentPer100Ms);
  return { observedPercent, assumedLossPercent, remainderPercentagePoints: round(observedPercent - assumedLossPercent),
    incrementalMs: assumptions.displayIncrementalMs, measuredNetRevenue: false as const };
}
