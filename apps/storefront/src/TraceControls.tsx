import { lazy, Suspense } from "react";
import { automaticTelemetryConfig } from "./session-trace-bridge";
import { TelemetryNotice } from "./TelemetryNotice";
const LegacyTraceControls = lazy(() => import("./LegacyTraceControls"));
/** The pinned store hook mounts this only for configured telemetry or local debug. */
export default function TraceControls() {
  return automaticTelemetryConfig ? <TelemetryNotice config={automaticTelemetryConfig} />
    : <Suspense fallback={null}><LegacyTraceControls /></Suspense>;
}
