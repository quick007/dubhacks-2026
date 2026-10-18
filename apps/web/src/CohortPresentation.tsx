import { useEffect, useState } from "react";
import { fetchPresentationResults, type PresentationResults } from "./lib/presentation-results";
import { Deck } from "./slides/Deck";
import { CohortResultsSlide } from "./slides/CohortResultsSlide";
import { SpeedSlide } from "./slides/SpeedSlide";
import { StakesSlide } from "./slides/StakesSlide";
import { TitleSlide } from "./slides/TitleSlide";
import { FlywheelSlide } from "./slides/FlywheelSlide";
import { TrustSlide } from "./slides/TrustSlide";
import { ShopperContextSlide } from "./slides/ShopperContextSlide";
import { DisclaimerSlide } from "./slides/DisclaimerSlide";

/** A small measured deck, backed by an explicitly selected immutable Tiger report. */
export function CohortPresentation({ reportId }: { reportId: string }) {
  const [result, setResult] = useState<PresentationResults>();
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setResult(undefined); setError("");
    void fetchPresentationResults(reportId, { signal: controller.signal }).then((value) => {
      if (!controller.signal.aborted) setResult(value);
    }).catch((cause: unknown) => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Results unavailable.");
    });
    return () => controller.abort();
  }, [reportId]);
  if (!result || result.reportId !== reportId) return <p role="status" className="p-8 text-lg text-muted">{error || "Loading verified results…"}</p>;
  return <Deck slides={[
    { id: "title", label: "Storefront Lab", render: () => <TitleSlide names={[]} /> },
    { id: "inputs", label: "Shopper context", render: () => <ShopperContextSlide /> },
    { id: "stakes", label: "Why speed matters", render: () => <StakesSlide /> },
    { id: "results", label: "Observed shopper results", render: () => <CohortResultsSlide result={result} /> },
    { id: "speed", label: "Ranking comparison", render: () => <SpeedSlide /> },
    { id: "flywheel", label: "The learning loop", render: () => <FlywheelSlide /> },
    { id: "trust", label: "Trust and responsibility", render: () => <TrustSlide /> },
    { id: "disclaimer", label: "Disclaimers", render: () => <DisclaimerSlide reportId={result.reportId} /> },
  ]} />;
}
