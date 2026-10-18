import { useEffect, useState } from "react";
import { privacyChoice, type ShoppingTelemetryConfig } from "./shopping-telemetry-config";
import type { startShoppingTelemetry } from "./shopping-telemetry-runtime";
const preferenceKey = "fieldwork.shopping-privacy.v1";
function preference() { try { return localStorage.getItem(preferenceKey); } catch { return null; } }
function signals() { return { globalPrivacyControl: (navigator as Navigator & { globalPrivacyControl?: boolean }).globalPrivacyControl, doNotTrack: navigator.doNotTrack }; }
export function TelemetryNotice({ config }: { config: ShoppingTelemetryConfig }) {
  const [choice, setChoice] = useState(() => privacyChoice(config, preference(), signals()));
  const [visit, setVisit] = useState(0);
  const allowed = choice === "notice" || choice === "granted";
  useEffect(() => {
    let disposed = false;
    let session: ReturnType<typeof startShoppingTelemetry> | undefined;
    if (allowed) void import("./shopping-telemetry-runtime").then(({ startShoppingTelemetry }) => {
      if (disposed) return;
      session = startShoppingTelemetry(config, choice as "notice" | "granted");
      window.fieldworkShopping = { stats: session.stats, flush: session.flush };
    });
    return () => { disposed = true; session?.stop(); delete window.fieldworkShopping; };
  }, [allowed, choice, config, visit]);
  useEffect(() => {
    const sync = () => setChoice(privacyChoice(config, preference(), signals()));
    const restored = (event: PageTransitionEvent) => { if (event.persisted) setVisit((value) => value + 1); };
    window.addEventListener("storage", sync); window.addEventListener("focus", sync); window.addEventListener("pageshow", restored);
    return () => { window.removeEventListener("storage", sync); window.removeEventListener("focus", sync); window.removeEventListener("pageshow", restored); };
  }, [config]);
  function choose(value: "granted" | "denied") {
    try { localStorage.setItem(preferenceKey, value); } catch { /* Choice still applies to this page. */ }
    setChoice(privacyChoice(config, value, signals()));
  }
  const blocked = !!signals().globalPrivacyControl || signals().doNotTrack === "1";
  return <div className="demo-controls trace-controls" data-trace-private>
    <button className="demo-button" popoverTarget="shopping-privacy">{allowed ? "Anonymous recording on" : "Shopping privacy"}</button>
    <div id="shopping-privacy" className="demo-popover" popover="auto" role="group" aria-label="Shopping privacy">
      <span className="demo-title">Shopping privacy</span>
      <p>We record anonymous product views, shopping actions and masked page replay to improve this shop. Text, form fields, account and payment areas are hidden. Data goes to our backend and Tiger Data; replay expires after 24 hours.</p>
      <p role="status">{allowed ? "Sharing is on. You can stop future collection at any time." : blocked ? "Your browser privacy preference has disabled sharing." : "Sharing is off until you allow it."}</p>
      <div className="theme-options">
        {!allowed && !blocked && <button onClick={() => choose("granted")}>Allow anonymous session recording</button>}
        <button onClick={() => choose("denied")}>{allowed ? "Stop sharing" : "Keep sharing off"}</button>
      </div>
    </div>
  </div>;
}
declare global { interface Window { fieldworkShopping?: { stats: () => unknown; flush: () => Promise<void> } } }
