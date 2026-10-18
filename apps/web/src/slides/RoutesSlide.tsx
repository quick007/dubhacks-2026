import type { Pair } from "../lib/story";
import { ShopMap } from "./ShopMap";

export function RoutesSlide({ pairs }: { pairs: Pair[] }) {
  return (
    <section className="mx-auto flex w-full max-w-6xl flex-col justify-center gap-8 px-4 py-6 sm:px-8">
      <h2 className="text-center font-display text-3xl leading-tight font-extrabold tracking-[-0.015em] sm:text-4xl">
        Every shopper's route
      </h2>
      <div className="grid w-full grid-cols-2 gap-6 self-center sm:w-[min(100%,calc((100dvh-22rem)*2))] sm:grid-cols-1 sm:gap-6">
        <ShopMap pairs={pairs} arm="baseline" />
        <div className="border-line sm:border-t sm:pt-6">
          <ShopMap pairs={pairs} arm="candidate" />
        </div>
      </div>
    </section>
  );
}
