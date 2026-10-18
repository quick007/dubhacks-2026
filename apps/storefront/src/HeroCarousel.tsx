import { useRef } from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import type { Product } from "./config";
import { HERO_BANNERS } from "./hero-banners";

/** One stable, factual editorial image selected from this person's learned order. */
export function HeroCarousel({ products, pending = false, unavailable = false }: { products: Product[]; pending?: boolean; unavailable?: boolean }) {
  const chosen = useRef<Product | null>(null);
  if (!pending && !unavailable && !chosen.current) chosen.current = products.find(product => HERO_BANNERS[product.id]) ?? null;
  const product = chosen.current;
  if (pending || unavailable || !product) return <div className={`hero-visual hero-pending${pending ? " skeleton-surface" : ""}`} aria-hidden="true" />;
  const available = products.some(item => item.id === product.id);
  const target = available ? `product-${product.id}` : "collection";
  return (
    <div className="hero-visual" role="region" aria-label="Featured for this shopping profile">
      <div className="hero-slide" data-active="true">
        <span className="hero-banner">
          <img src={`/products/hero-${product.id}-wide.jpg`} alt={HERO_BANNERS[product.id]![0]}
            width={2048} height={2048} loading="eager" style={{ top: "0%" }} />
        </span>
      </div>
      <a className="hero-product-link" href={`#${target}`} onClick={event => {
        event.preventDefault();
        document.getElementById(target)?.scrollIntoView({ behavior: "smooth", block: "start" });
      }}>
        <span>{available ? `Shop ${product.name}` : "Shop the collection"}</span>
        <HugeiconsIcon icon={ArrowRight01Icon} size={22} aria-hidden="true" />
      </a>
    </div>
  );
}
