import { HugeiconsIcon } from "@hugeicons/react";
import { Search01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import type { Product, VariantConfig } from "./config";
import type { personalizeStorefront } from "./personalization";
import { COLLECTIONS } from "./collections";
import { ProductCard } from "./ProductCard";
import { HeroCarousel } from "./HeroCarousel";
import { CategoryShortcuts } from "./CategoryShortcuts";

export function ShopView({
  config,
  experience,
  collectionId,
  query,
  searchInput,
  visibleProducts,
  selectedVariants,
  onSearchInput,
  onSearchSubmit,
  onClearSearch,
  onCollectionChange,
  onVariantChange,
  onAdd,
}: {
  config: VariantConfig;
  experience: ReturnType<typeof personalizeStorefront>;
  collectionId: string;
  query: string;
  searchInput: string;
  visibleProducts: Product[];
  selectedVariants: Partial<Record<Product["id"], string>>;
  onSearchInput: (value: string) => void;
  onSearchSubmit: () => void;
  onClearSearch: () => void;
  onCollectionChange: (collectionId: string) => void;
  onVariantChange: (productId: Product["id"], variantId: string) => void;
  onAdd: (product: Product, variantId: string, colorName: string) => void;
}) {
  const collection = COLLECTIONS.find((item) => item.id === collectionId) ?? COLLECTIONS[0];
  const heading = collection.id === "all" ? experience.collection : collection.label;

  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <h1 tabIndex={-1}>{experience.headline}</h1>
          <a
            className="hero-shop-link"
            href="#collection"
            onClick={(event) => {
              event.preventDefault();
              document
                .getElementById("collection")
                ?.scrollIntoView({ behavior: "smooth", block: "start" });
            }}
          >
            Shop the collection
            <HugeiconsIcon icon={ArrowRight01Icon} size={20} aria-hidden="true" />
          </a>
        </div>
        <HeroCarousel key={`${collectionId}:${query}`} products={visibleProducts} />
      </section>
      <CategoryShortcuts collectionId={collectionId} onChange={onCollectionChange} />
      {config.showShipping && (
        <div className="shipping-info">
          Free demo shipping on orders over $50 · $5 flat rate otherwise
        </div>
      )}
      <section id="collection" className="collection">
        <div className="section-heading">
          <div>
            <h2>{heading}</h2>
            <p className="search-status" role="status">
              {query &&
                (experience.matched
                  ? `Picked for ${experience.reason ?? `“${query}”`}`
                  : "No close matches. Explore the collection below.")}
            </p>
          </div>
          <div className="collection-search">
            <form
              className="shop-search"
              role="search"
              onSubmit={(event) => {
                event.preventDefault();
                onSearchSubmit();
              }}
            >
              <label className="search-label" htmlFor="shop-search">
                Search the collection
              </label>
              <div className="search-field">
                <HugeiconsIcon icon={Search01Icon} size={22} aria-hidden="true" />
                <input
                  id="shop-search"
                  type="search"
                  placeholder="Search the collection"
                  value={searchInput}
                  maxLength={200}
                  onChange={(event) => onSearchInput(event.target.value)}
                />
                <button type="submit">Search</button>
              </div>
            </form>
            {query && (
              <button className="clear-search" onClick={onClearSearch}>
                Clear search
              </button>
            )}
          </div>
        </div>
        <div className="products">
          {visibleProducts.map((product) => (
            <ProductCard
              key={product.id}
              product={product}
              context={experience.context}
              variantId={selectedVariants[product.id]}
              onVariantChange={(variantId) => onVariantChange(product.id, variantId)}
              onAdd={(variantId, colorName) => onAdd(product, variantId, colorName)}
            />
          ))}
        </div>
      </section>
    </>
  );
}
