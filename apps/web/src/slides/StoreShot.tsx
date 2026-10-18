import { Check, X } from "lucide-react";
import stages from "../fieldwork/title-stages.json";
import { modelRequest } from "../lib/model";
import { ShopperCursor } from "./ShopperCursor";

export type ShotStage = "browse" | "stuck" | "fit" | "second";
type Card = (typeof stages.browse.cards)[number];

/** A card's part that is inside the screenshot, as CSS percentages. */
function box(card: Card) {
  const top = Math.max(card.y, 0);
  const bottom = Math.min(card.y + card.height, 1);
  return {
    left: `${card.x * 100}%`,
    top: `${top * 100}%`,
    width: `${card.width * 100}%`,
    height: `${(bottom - top) * 100}%`,
  };
}

/**
 * The real demo shop, captured by scripts/capture-fieldwork.mjs: shoppers browsing it, the featured
 * shopper leaving with the product one scroll away, then the same shopper's page in the model's order.
 * "second" is another shopper's page in the model's order. `names` starts with the shopper shown.
 */
export function StoreShot({ stage, names }: { stage: ShotStage; names: string[] }) {
  const shot = stages[stage];
  const targetId = stage === "second" ? modelRequest(stages.secondShopper)?.target : stages.target;
  const target = shot.cards.find((card) => card.id === targetId);
  return (
    <div
      aria-hidden
      className="relative aspect-[8/5] overflow-hidden rounded-lg border border-line bg-paper"
    >
      <img src={shot.src} alt="" className="size-full object-cover" />
      {stage === "browse" &&
        shot.cards
          .slice(0, 3)
          .map((card, i) => (
            <ShopperCursor
              key={card.id}
              name={names[i]}
              index={i}
              at={{ x: 0.12 + i * 0.24, y: 0.28 + i * 0.13 }}
              via={{ x: 0.6 - i * 0.22, y: 0.4 + i * 0.07 }}
              to={{ x: 0.32 + i * 0.14, y: 0.56 - i * 0.14 }}
            />
          ))}
      {stage === "stuck" && target && (
        <>
          <span
            className="absolute rounded-md border-2 border-dashed border-signal"
            style={box(target)}
          />
          <ShopperCursor
            name={names[0]}
            at={{ x: 0.18, y: 0.12 }}
            via={{ x: 0.6, y: 0.17 }}
            to={{ x: 0.42, y: 0.22 }}
          />
          <span className="absolute top-[58%] left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full bg-signal px-2 py-0.5 text-xs font-bold whitespace-nowrap text-paper max-md:px-1">
            <X className="size-3" strokeWidth={3} />
            <span className="max-md:hidden">Left one scroll short</span>
          </span>
        </>
      )}
      {(stage === "fit" || stage === "second") && target && (
        <span className="absolute rounded-md ring-4 ring-cobalt" style={box(target)}>
          <span className="absolute -top-3 left-2 flex items-center gap-1 rounded-full bg-cobalt px-2 py-0.5 text-xs font-bold whitespace-nowrap text-paper max-md:px-1">
            <Check className="size-3" strokeWidth={3} />
            <span className="max-md:hidden">{names[0]}</span>
          </span>
        </span>
      )}
    </div>
  );
}
