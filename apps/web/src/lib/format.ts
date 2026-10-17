export function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}
export function splitLabel(value: "development" | "held-out") {
  return value === "development" ? "Development" : "Held-out";
}
