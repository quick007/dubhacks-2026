import { HugeiconsIcon } from "@hugeicons/react";
import { Search01Icon } from "@hugeicons/core-free-icons";
import type { Product, VariantConfig } from "./config";
import type { personalizeStorefront } from "./personalization";
import { COLLECTIONS } from "./collections";
import { ProductCard } from "./ProductCard";
import { localModelStatus, type LocalModelState } from "./local-model";

const EXAMPLE_SEARCHES = [
  "Rain jacket for campus",
  "Work bag",
  "Family park",
  "Hiking gear",
];

export function ShopView({
  config,
  experience,
  collectionId,
  query,
  searchInput,
  visibleProducts,
  selectedVariants,
  added,
  localModel,
  onSearchInput,
  onSearchSubmit,
  onExampleSearch,
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
  localModel?: LocalModelState;
  onSearchInput: (value: string) => void;
  onSearchSubmit: () => void;
  onExampleSearch: (query: string) => void;
  onClearSearch: () => void;
  onCollectionChange: (collectionId: string) => void;
  onVariantChange: (productId: Product["id"], variantId: string) => void;
  onAdd: (product: Product, variantId: string, colorName: string) => void;
}) {
  const collection = COLLECTIONS.find((item) => item.id === collectionId) ?? COLLECTIONS[0];
  const heading = collection.id === "all"
    ? localModel?.copy?.sections[0]?.heading ?? experience.collection
    : collection.label;
  const localCopy = new Map(localModel?.copy?.sections.flatMap((section) => section.cards).map((card) => [card.productId, card]));
  return (
    <>
      <section className="hero">
        <h1>{experience.headline}</h1>
        <form
          className="shop-search"
          role="search"
          onSubmit={(event) => {
            event.preventDefault();
            onSearchSubmit();
          }}
        >
          <label htmlFor="shop-search">What are you looking for?</label>
          <div className="search-field">
            <HugeiconsIcon icon={Search01Icon} size={22} aria-hidden="true" />
            <input
              id="shop-search"
              type="search"
              placeholder="Try a rain jacket for campus or a work bag"
              value={searchInput}
              maxLength={200}
              onChange={(event) => onSearchInput(event.target.value)}
            />
            <button type="submit">Search</button>
          </div>
        </form>
        <div className="example-searches" role="group" aria-label="Example searches">
          <span className="example-label" aria-hidden="true">
            Try
          </span>
          {EXAMPLE_SEARCHES.map((example) => (
            <button type="button" key={example} onClick={() => onExampleSearch(example)}>
              {example}
            </button>
          ))}
        </div>
        {config.showShipping && (
          <div className="shipping-info">
            Free demo shipping on orders over $50 · $5 flat rate otherwise
          </div>
        )}
        <nav className="collection-options" aria-label="Shop by category">
          {COLLECTIONS.map((item) => (
            <button
              type="button"
              key={item.id}
              aria-pressed={collectionId === item.id}
              onClick={() => onCollectionChange(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </section>
      <section id="collection" className="collection">
        <div className="section-heading">
          <div>
            <h2>{heading}</h2>
            <p className="search-status" role="status">
              {localModel ? localModelStatus(localModel) : query &&
                (experience.matched
                  ? `Suggestions for “${query}”${experience.reason ? ` — ordered for ${experience.reason}` : ""}`
                  : `No close matches for “${query}”. Browse the collection below.`)}
            </p>
          </div>
          {query && (
            <button className="clear-search" onClick={onClearSearch}>
              Clear search
            </button>
          )}
        </div>
        <div className="products">
          {visibleProducts.map((product) => (
            <ProductCard
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
        <p className="added-message" role="status">
          {added}
        </p>
      </section>
    </>
  );
}
