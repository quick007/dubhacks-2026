import type { Pair } from "../lib/story";
import { MerchantValue } from "./MerchantValue";
import { SlideFrame } from "./SlideFrame";

export function StoreSlide({ pairs }: { pairs: Pair[] }) {
  return (
    <SlideFrame
      title="What the store gets"
      lead="The same new shoppers, counted from the store's side. Only purchases that meet the shopper's needs count as value."
    >
      <MerchantValue pairs={pairs} />
    </SlideFrame>
  );
}
