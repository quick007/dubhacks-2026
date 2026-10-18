import { useEffect, useState } from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowLeft01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import type { Product } from "./config";
import { selectShowcaseImage, type ImageContext } from "./images";
import { ProductPhoto } from "./ProductPhoto";

export function HeroCarousel({ products, context }: { products: Product[]; context: ImageContext }) {
  const slides = products.slice(0, 5).map((product) => ({
    product,
    image: selectShowcaseImage(product.id, context),
  }));
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const count = slides.length;

  useEffect(() => {
    if (paused || hovered || focused || count < 2) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) setIndex((current) => (current + 1) % count);
    }, 6000);
    return () => window.clearInterval(timer);
  }, [paused, hovered, focused, count]);

  const slide = slides[index % count];
  if (!slide) return null;

  function move(delta: number) {
    setPaused(true);
    setIndex((current) => (current + delta + count) % count);
  }

  return (
    <div
      className="hero-visual"
      role="region"
      aria-roledescription="carousel"
      aria-label="Featured products"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onFocusCapture={() => setFocused(true)}
      onBlurCapture={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setFocused(false);
      }}
    >
      <ProductPhoto image={slide.image} loading="eager" />
      <a
        className="hero-product-link"
        href={`#product-${slide.product.id}`}
        onClick={(event) => {
          event.preventDefault();
          document.getElementById(`product-${slide.product.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
        }}
      >
        <span>{slide.product.name}</span>
        <HugeiconsIcon icon={ArrowRight01Icon} size={22} aria-hidden="true" />
      </a>
      {count > 1 && (
        <div className="hero-slide-controls">
          <button type="button" aria-label="Previous featured photo" onClick={() => move(-1)}>
            <HugeiconsIcon icon={ArrowLeft01Icon} size={18} aria-hidden="true" />
          </button>
          <span aria-live={paused ? "polite" : "off"}>{index + 1} / {count}</span>
          <button type="button" aria-label="Next featured photo" onClick={() => move(1)}>
            <HugeiconsIcon icon={ArrowRight01Icon} size={18} aria-hidden="true" />
          </button>
          <button type="button" onClick={() => setPaused((value) => !value)} aria-label={paused ? "Play featured slideshow" : "Pause featured slideshow"}>
            {paused ? "Play" : "Pause"}
          </button>
        </div>
      )}
    </div>
  );
}
