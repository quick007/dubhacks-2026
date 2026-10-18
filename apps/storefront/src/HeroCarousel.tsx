import { useEffect, useState } from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import type { Product } from "./config";
import { HERO_BANNERS } from "./hero-banners";

export function HeroCarousel({ products }: { products: Product[] }) {
  const featured = products.filter((product) => HERO_BANNERS[product.id]).slice(0, 4);
  const slides = [0, 1].flatMap((row) =>
    featured.map((product) => ({
      product,
      row,
      alt: HERO_BANNERS[product.id]![row],
    })),
  );
  const [index, setIndex] = useState(0);
  const count = slides.length;

  useEffect(() => {
    if (count < 2) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) setIndex((current) => (current + 1) % count);
    }, 6000);
    return () => window.clearInterval(timer);
  }, [count]);

  const slide = slides[index % count];
  if (!slide) return null;

  return (
    <div
      className="hero-visual"
      role="region"
      aria-roledescription="carousel"
      aria-label="Featured products"
    >
      {slides.map((item, slideIndex) => (
        <div
          className="hero-slide"
          key={`${item.product.id}:${item.row}`}
          data-active={slideIndex === index % count}
          aria-hidden={slideIndex !== index % count}
        >
          <span className="hero-banner">
            <img
              src={`/products/hero-${item.product.id}-wide.jpg`}
              alt={item.alt}
              width={2048}
              height={2048}
              loading={slideIndex === 0 ? "eager" : "lazy"}
              style={{ top: `${-item.row * 100}%` }}
            />
          </span>
        </div>
      ))}
      <a
        className="hero-product-link"
        href={`#product-${slide.product.id}`}
        onClick={(event) => {
          event.preventDefault();
          document
            .getElementById(`product-${slide.product.id}`)
            ?.scrollIntoView({ behavior: "smooth", block: "start" });
        }}
      >
        <span>Shop {slide.product.name}</span>
        <HugeiconsIcon icon={ArrowRight01Icon} size={22} aria-hidden="true" />
      </a>
    </div>
  );
}
