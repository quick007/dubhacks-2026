import { useEffect, useState } from "react";
import { startTrace, stopTrace, clearTrace, exportTrace, replayTrace, traceStatus } from "./session-trace-runtime";

/** Only mounted for ?trace=demo. Recording still requires an explicit Start. */
export default function TraceControls() {
  const [status, setStatus] = useState(traceStatus());
  useEffect(() => { const sync = () => setStatus(traceStatus()); window.addEventListener("fieldwork-trace-status", sync); return () => { window.removeEventListener("fieldwork-trace-status", sync); clearTrace(); }; }, []);
  return <div className="demo-controls trace-controls" data-trace-private>
    <button className="demo-button" popoverTarget="trace-controls">Session trace</button>
    <div id="trace-controls" className="demo-popover" popover="auto" role="group" aria-label="Session trace">
      <span className="demo-title">Synthetic demo sessions only</span>
      <p role="status">{status}. Local memory only. Text and inputs hidden.</p>
      <div className="theme-options">
        <button onClick={() => void startTrace()} disabled={status === "Recording" || status === "Loading"}>Start recording</button>
        <button onClick={() => stopTrace()} disabled={status !== "Recording" && status !== "Loading"}>Stop</button>
        <button onClick={() => clearTrace()}>Clear</button>
        <button onClick={() => exportTrace()}>Export JSON</button>
        <button onClick={() => void replayTrace()}>Replay</button>
      </div>
    </div>
  </div>;
}
