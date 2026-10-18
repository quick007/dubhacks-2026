import React, { useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { HugeiconsIcon } from "@hugeicons/react";
import { ShoppingCart01Icon } from "@hugeicons/core-free-icons";
import {
  parseVariantConfig,
  serializeVariantConfig,
  type Product,
  type VariantConfig,
} from "./config";
import {
  addCartItem,
  getCartLines,
  removeCartLine,
  removeOneCartItem,
  type CartItem,
} from "./cart";
import { personalizeStorefront } from "./personalization";
import { COLLECTIONS } from "./collections";
import { orderTotal, shippingQuote } from "./shipping";
import { DemoControls } from "./DemoControls";
import { ShopView } from "./ShopView";
import { CartView } from "./CartView";
import { ConfirmationView } from "./ConfirmationView";
import { useRoute } from "./use-route";
import type { StorefrontRoute } from "./routes";
import "./styles.css";

const initialConfig = parseVariantConfig(window.location.search);

function App() {
  const { route, navigate } = useRoute();
  const [config, setConfig] = useState(initialConfig);
  const [cart, setCart] = useState<CartItem[]>([]);
  const [selectedVariants, setSelectedVariants] = useState<Partial<Record<Product["id"], string>>>(
    {},
  );
  const [added, setAdded] = useState<{ id: number; text: string } | null>(null);
  const [searchInput, setSearchInput] = useState(route.query);

  useEffect(() => {
    setSearchInput(route.query);
  }, [route.query]);

  const experience = personalizeStorefront(route.query, config);
  const collection = COLLECTIONS.find((item) => item.id === route.collectionId) ?? COLLECTIONS[0];
  const visibleProducts =
    collection.id === "all"
      ? experience.products
      : experience.products.filter((product) => collection.productIds.includes(product.id));
  const cartLines = getCartLines(cart);
  const subtotal = cartLines.reduce((sum, line) => sum + line.subtotal, 0);
  const quote = shippingQuote(subtotal, config.showShipping);
  const total = orderTotal(subtotal, quote);
  // A shared or restored #confirmed link without cart contents falls back to the cart.
  const page = route.page === "confirmed" && cart.length === 0 ? "cart" : route.page;

  useEffect(() => {
    document.title =
      page === "cart"
        ? "Fieldwork — Demo cart"
        : page === "confirmed"
          ? "Fieldwork — Demo order"
          : route.query
            ? `Fieldwork — ${route.query}`
            : "Fieldwork — Demo Shop";
  }, [page, route.query]);

  // Dismiss the add-to-cart confirmation after a short pause.
  useEffect(() => {
    if (!added) return;
    const timer = window.setTimeout(() => setAdded(null), 4000);
    return () => window.clearTimeout(timer);
  }, [added]);

  useEffect(() => {
    if (page !== "shop") setAdded(null);
  }, [page]);

  // Move focus to the new view heading so screen readers announce the change.
  const previousPage = useRef(page);
  useEffect(() => {
    if (previousPage.current === page) return;
    previousPage.current = page;
    document.querySelector<HTMLElement>("main h1")?.focus({ preventScroll: true });
  }, [page]);

  function goToCollection(
    next: Partial<Pick<StorefrontRoute, "query" | "collectionId">>,
    options?: { replace?: boolean; scrollTo?: "hero" | "collection" },
  ) {
    navigate(
      {
        page: "shop",
        query: next.query ?? route.query,
        collectionId: next.collectionId ?? route.collectionId,
      },
      options,
    );
    const target =
      options?.scrollTo === "hero"
        ? document.querySelector(".hero")
        : options?.scrollTo === "collection"
          ? document.getElementById("collection")
          : null;
    target?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function applyConfig(next: VariantConfig) {
    setConfig(next);
    const search = serializeVariantConfig(next);
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${search ? `?${search}` : ""}${window.location.hash}`,
    );
  }

  return (
    <>
      <div className="demo-banner">Demo store · No real orders or payments</div>
      <header>
        <a
          className="wordmark"
          href="#collection"
          onClick={(event) => {
            event.preventDefault();
            goToCollection({ query: "", collectionId: "all" });
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        >
          fieldwork<span>®</span>
        </a>
        <div className="header-actions">
          <DemoControls config={config} onChange={applyConfig} />
          <button
            aria-label="View cart"
            className="cart-button"
            onClick={() => navigate({ ...route, page: "cart" })}
          >
            <HugeiconsIcon icon={ShoppingCart01Icon} size={24} aria-hidden="true" />
            <span className="cart-button-label">View cart</span>
            {cart.length > 0 && <span className="cart-count">{cart.length}</span>}
          </button>
        </div>
      </header>
      <main>
        {page === "shop" && (
          <ShopView
            config={config}
            experience={experience}
            collectionId={route.collectionId}
            query={route.query}
            searchInput={searchInput}
            visibleProducts={visibleProducts}
            selectedVariants={selectedVariants}
            onSearchInput={(value) => {
              setSearchInput(value);
              if (!value.trim() && route.query) goToCollection({ query: "" });
            }}
            onSearchSubmit={() => {
              goToCollection({ query: searchInput.trim() }, { scrollTo: "hero" });
            }}
            onExampleSearch={(query) => {
              setSearchInput(query);
              goToCollection({ query }, { scrollTo: "hero" });
            }}
            onClearSearch={() => {
              setSearchInput("");
              goToCollection({ query: "" });
              document.getElementById("shop-search")?.focus();
            }}
            onCollectionChange={(collectionId) =>
              goToCollection({ collectionId }, { scrollTo: "collection" })
            }
            onVariantChange={(productId, variantId) =>
              setSelectedVariants((items) => ({ ...items, [productId]: variantId }))
            }
            onAdd={(product, variantId, colorName) => {
              setCart((items) => addCartItem(items, { productId: product.id, variantId }));
              setAdded((current) => ({
                id: (current?.id ?? 0) + 1,
                text: `${product.name} (${colorName}) added to cart.`,
              }));
            }}
          />
        )}
        {page === "cart" && (
          <CartView
            lines={cartLines}
            subtotal={subtotal}
            quote={quote}
            total={total}
            onQuantity={(variantId, change) =>
              setCart((items) => {
                if (change === 1) {
                  const line = cartLines.find((candidate) => candidate.variant.id === variantId);
                  return line ? addCartItem(items, { productId: line.product.id, variantId }) : items;
                }
                return removeOneCartItem(items, variantId);
              })
            }
            onRemove={(variantId) => setCart((items) => removeCartLine(items, variantId))}
            onPlaceOrder={() => {
              navigate({ ...route, page: "confirmed" });
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            onContinue={() => goToCollection({})}
          />
        )}
        {page === "confirmed" && (
          <ConfirmationView
            itemCount={cart.length}
            subtotal={subtotal}
            quote={quote}
            total={total}
            onBackToCollection={() => {
              setCart([]);
              setAdded(null);
              goToCollection({}, { replace: true });
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          />
        )}
      </main>
      {page === "shop" && added && (
        <div className="add-toast" role="status">
          <span>{added.text}</span>
          <button type="button" onClick={() => navigate({ ...route, page: "cart" })}>
            Open cart
          </button>
        </div>
      )}
      <footer>
        <span className="wordmark">
          fieldwork<span>®</span>
        </span>
      </footer>
    </>
  );
}
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
