import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import type { Product, VariantConfig } from "./config";
import type { personalizeStorefront } from "./personalization";
import { COLLECTIONS } from "./collections";
import { ProductCard } from "./ProductCard";
import { HeroCarousel } from "./HeroCarousel";
import { CategoryShortcuts } from "./CategoryShortcuts";
import { ProductSkeleton } from "./ProductSkeleton";
import type { LocalModelState } from "./local-model";

export function ShopView({
  config,
  experience,
  collectionId,
  query,
  searchInput,
  visibleProducts,
  selectedVariants,
  localModel,
  personName,
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
  added: string;
  onExampleSearch: (query: string) => void;
  localModel?: LocalModelState;
  personName?: string;
  onSearchInput: (value: string) => void;
  onSearchSubmit: () => void;
  onClearSearch: () => void;
  onCollectionChange: (collectionId: string) => void;
  onVariantChange: (productId: Product["id"], variantId: string) => void;
  onAdd: (product: Product, variantId: string, colorName: string) => void;
}) {
  const collection = COLLECTIONS.find((item) => item.id === collectionId) ?? COLLECTIONS[0];
  const failed = localModel?.phase === "error";
  const loading = Boolean(localModel && !failed && !localModel.copy);
  const heading = collection.id === "all"
    ? (!failed && localModel?.copy?.sections[0]?.heading) || experience.collection
    : collection.label;
  const localCopy = new Map(localModel?.copy?.sections.flatMap(section => section.cards).map(card => [card.productId, card]));

  return (
    <>
      <section className="hero" hidden={Boolean(query)}>
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
        <HeroCarousel key={personName ?? "baseline"} products={visibleProducts} pending={loading} unavailable={failed} />
      </section>
      <CategoryShortcuts collectionId={collectionId} onChange={onCollectionChange} />
      {config.showShipping && (
        <div className="shipping-info">
          Free shipping on orders over $50 · $5 flat rate otherwise
        </div>
      )}
      <section id="collection" className="collection" aria-busy={loading || localModel?.phase === "partial"}>
        <div className="section-heading">
          <div>
            <h2>{heading}</h2>
            <p className="search-status" role="status">
              {localModel ? (failed ? "We couldn’t refresh your collection. Try another search."
                : loading ? `Finding a few things for ${personName ?? "you"}…`
                : localModel.phase === "partial" ? "Finishing your collection…"
                : visibleProducts.length === 0 ? "No items match this request. Try broadening your search."
                : query ? `Suggestions for “${query}”` : `A collection for ${personName ?? "you"}`)
                : query && (experience.matched ? `Picked for ${experience.reason ?? `“${query}”`}`
                  : "No close matches. Explore the collection below.")}
            </p>
          </div>
          {query && <button className="clear-search" onClick={onClearSearch}>Clear search</button>}
        </div>
        <div className="products">
          {loading ? Array.from({ length: 6 }, (_, index) => <ProductSkeleton key={index} />)
            : !failed && visibleProducts.map((product) => (
            localModel?.rank?.selectedProductIds.includes(product.id) && !localCopy.has(product.id)
              ? <ProductSkeleton key={product.id} /> : <ProductCard
              key={product.id}
              product={product}
              context={experience.context}
              variantId={selectedVariants[product.id] ?? localModel?.rank?.variants[product.id]}
              localCopy={localCopy.get(product.id)}
              onVariantChange={(variantId) => onVariantChange(product.id, variantId)}
              onAdd={(variantId, colorName) => onAdd(product, variantId, colorName)}
            />
          ))}
        </div>
      </section>
    </>
  );
}
