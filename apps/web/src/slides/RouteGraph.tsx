import { useId } from "react";
import type { flows, PageKind } from "../lib/story";

type Station = { x: number; y: number; label: string };
type Flow = ReturnType<typeof flows>;
const horizontal: Partial<Record<PageKind, Station>> = {
  home: { x: 45, y: 120, label: "Home" },
  category: { x: 220, y: 55, label: "Category" },
  search: { x: 220, y: 185, label: "Search" },
  product: { x: 410, y: 120, label: "Product" },
  cart: { x: 580, y: 120, label: "Cart" },
  checkout: { x: 740, y: 120, label: "Checkout" },
  confirmation: { x: 910, y: 120, label: "Bought" },
};
const vertical: Partial<Record<PageKind, Station>> = {
  home: { x: 80, y: 30, label: "Home" },
  category: { x: 30, y: 110, label: "Category" },
  search: { x: 130, y: 110, label: "Search" },
  product: { x: 80, y: 210, label: "Product" },
  cart: { x: 80, y: 290, label: "Cart" },
  checkout: { x: 80, y: 370, label: "Checkout" },
  confirmation: { x: 80, y: 450, label: "Bought" },
};

export function RouteGraph({
  flow,
  color,
  mobile = false,
}: {
  flow: Flow;
  color: string;
  mobile?: boolean;
}) {
  const marker = useId();
  const stations = mobile ? vertical : horizontal;
  return (
    <svg
      viewBox={mobile ? "0 0 180 490" : "0 0 960 240"}
      className={mobile ? "w-full sm:hidden" : "hidden w-full sm:block"}
      role="img"
      aria-label="Shopper paths from home to purchase, with return paths dashed and stops marked in red"
    >
      <defs>
        <marker
          id={marker}
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="4"
          markerHeight="4"
          orient="auto-start-reverse"
        >
          <path d="M1 1 L9 5 L1 9" fill="none" stroke={color} strokeWidth="2" />
        </marker>
      </defs>
      {flow.edges.map((edge) => {
        const from = stations[edge.from];
        const to = stations[edge.to];
        if (!from || !to) return null;
        const returning = mobile ? to.y < from.y : to.x < from.x;
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const length = Math.hypot(dx, dy) || 1;
        let path = `M${from.x + (dx / length) * 10},${from.y + (dy / length) * 10} L${to.x - (dx / length) * 14},${to.y - (dy / length) * 14}`;
        if (returning) {
          if (mobile) {
            const lane = to.x < from.x ? 8 : 158;
            path = `M${from.x + (lane < from.x ? -10 : 10)},${from.y} H${lane} V${to.y} H${to.x + (lane < to.x ? -14 : 14)}`;
          } else {
            const lane = to.y < from.y ? 14 : 226;
            path = `M${from.x},${from.y + (lane < from.y ? -10 : 10)} V${lane} H${to.x} V${to.y + (lane < to.y ? -14 : 14)}`;
          }
        }
        return (
          <path
            key={`${edge.from}-${edge.to}`}
            d={path}
            fill="none"
            stroke={color}
            strokeWidth={returning ? 1.5 : 3}
            strokeDasharray={returning ? "5 5" : undefined}
            strokeLinejoin="round"
            markerEnd={`url(#${marker})`}
          >
            <title>
              {from.label} to {to.label}: {edge.count} shoppers
            </title>
          </path>
        );
      })}
      {Object.entries(stations).map(([kind, station]) => {
        const stopped = flow.giveUps.get(kind as PageKind);
        const branch = kind === "category" || kind === "search";
        return (
          <g key={kind}>
            <circle
              cx={station.x}
              cy={station.y}
              r={8}
              fill="var(--color-fog)"
              stroke={stopped ? "var(--color-signal)" : color}
              strokeWidth={2.5}
            />
            <text
              x={mobile && !branch ? station.x + 17 : station.x}
              y={mobile && !branch ? station.y + 5 : station.y + 29}
              textAnchor={mobile && !branch ? "start" : "middle"}
              fill="var(--color-ink)"
              stroke="var(--color-fog)"
              strokeWidth={6}
              paintOrder="stroke"
              strokeLinejoin="round"
              fontSize={mobile ? 12 : 18}
              fontWeight={500}
            >
              {station.label}
            </text>
            {stopped && (
              <text
                x={station.x + 17}
                y={station.y - 17}
                fill="var(--color-signal)"
                fontSize={mobile ? 11 : 16}
                fontWeight={600}
              >
                {stopped} stopped
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
