import { latencySeries } from "../lib/latency";
import { LatencyChart } from "./LatencyChart";
import { SlideFrame } from "./SlideFrame";

export function SpeedSlide() {
  const series = latencySeries();
  return (
    <SlideFrame
      className="max-w-7xl py-6"
      title="Personalization without the wait"
    >
      <LatencyChart series={series} />
    </SlideFrame>
  );
}
