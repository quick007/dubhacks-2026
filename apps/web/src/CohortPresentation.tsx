import { useEffect, useState } from "react";
import { fetchPresentationResults, type PresentationResults } from "./lib/presentation-results";
import { Deck } from "./slides/Deck";
import { CohortResultsSlide } from "./slides/CohortResultsSlide";
import { ModelSlide } from "./slides/ModelSlide";
import { SpeedSlide } from "./slides/SpeedSlide";
import { StakesSlide } from "./slides/StakesSlide";
import { TitleSlide } from "./slides/TitleSlide";

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
  if (!result) return <p role="status" className="p-8 text-lg text-muted">{error || "Loading verified results…"}</p>;
  return <Deck slides={[
    { id: "title", label: "Storefront Lab", render: () => <TitleSlide names={[]} /> },
    { id: "results", label: "Shopper results", render: () => <CohortResultsSlide result={result} /> },
    { id: "model", label: "The model", render: () => <ModelSlide /> },
    { id: "speed", label: "Ranking comparison", render: () => <SpeedSlide /> },
    { id: "stakes", label: "Why speed matters", render: () => <StakesSlide /> },
  ]} />;
}
