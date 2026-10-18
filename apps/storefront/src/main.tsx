import React, { lazy, Suspense, useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { HugeiconsIcon } from "@hugeicons/react";
import { ShoppingCart01Icon } from "@hugeicons/core-free-icons";
import {
  parseVariantConfig,
  serializeVariantConfig,
  orderedProducts,
  type Product,
  type VariantConfig,
} from "./config";
import { getCartLines, type CartItem } from "./cart";
import { personalizeStorefront } from "./personalization";
import { COLLECTIONS } from "./collections";
import { orderTotal, shippingQuote } from "./shipping";
import { DemoControls } from "./DemoControls";
import { ShopView } from "./ShopView";
import { CartView } from "./CartView";
import { ConfirmationView } from "./ConfirmationView";
import { useRoute } from "./use-route";
import type { StorefrontRoute } from "./routes";
import { localModelEnabled, preserveLocalModelSearch } from "./local-model";
import { useLocalModel } from "./use-local-model";
import { demoTraceEnabled, traceContext, traceEvent, traceProducts } from "./session-trace-bridge";
import "./styles.css";

const TraceControls = lazy(() => import("./TraceControls"));

const initialConfig = parseVariantConfig(window.location.search);

function App() {
  const { route, navigate } = useRoute();
  const [config, setConfig] = useState(initialConfig);
  const [cart, setCart] = useState<CartItem[]>([]);
  const [selectedVariants, setSelectedVariants] = useState<Partial<Record<Product["id"], string>>>(
    {},
  );
  const [added, setAdded] = useState("");
  const [searchInput, setSearchInput] = useState(route.query);

  useEffect(() => {
    setSearchInput(route.query);
  }, [route.query]);

  const localMode = localModelEnabled(window.location.search);
  const localModel = useLocalModel(localMode && route.page === "shop", route.query);
  const defaultExperience = personalizeStorefront(route.query, config);
  const catalogProducts = orderedProducts(config.recommendationOrder);
  const experience = localMode
    ? { ...defaultExperience, headline: config.headline, collection: "Collection", reason: null,
        matched: Boolean(localModel.rank),
        products: localModel.rank
          ? localModel.rank.productIds.flatMap((id) => catalogProducts.filter((product) => product.id === id))
          : catalogProducts }
    : defaultExperience;
  const collection = COLLECTIONS.find((item) => item.id === route.collectionId) ?? COLLECTIONS[0];
  const visibleProducts =
    collection.id === "all"
      ? experience.products
      : experience.products.filter((product) => collection.productIds.includes(product.id));
  useEffect(() => {
    traceContext({ mode: localMode ? "local-model" : "baseline", order: config.recommendationOrder, shipping: config.showShipping });
    traceProducts(route.page === "shop" ? visibleProducts.map((product) => product.id) : []);
  }, [localMode, config.recommendationOrder, config.showShipping, route.page, visibleProducts.map((product) => product.id).join(",")]);
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
    const search = preserveLocalModelSearch(serializeVariantConfig(next), window.location.search);
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
          {demoTraceEnabled && <Suspense fallback={null}><TraceControls /></Suspense>}
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
            added={added}
            localModel={localMode ? localModel : undefined}
            onSearchInput={(value) => {
              setSearchInput(value);
              if (!value.trim() && route.query) goToCollection({ query: "" });
            }}
            onSearchSubmit={() => {
              traceEvent("search", { source: "submit" });
              goToCollection({ query: searchInput.trim() }, { scrollTo: "hero" });
            }}
            onExampleSearch={(query) => {
              traceEvent("search", { source: "example" });
              setSearchInput(query);
              goToCollection({ query }, { scrollTo: "hero" });
            }}
            onClearSearch={() => {
              traceEvent("search", { source: "clear" });
              setSearchInput("");
              goToCollection({ query: "" });
              document.getElementById("shop-search")?.focus();
            }}
            onCollectionChange={(collectionId) =>
              goToCollection({ collectionId }, { scrollTo: "collection" })
            }
            onVariantChange={(productId, variantId) => {
              traceEvent("selection", { productId, variantId });
              setSelectedVariants((items) => ({ ...items, [productId]: variantId }));
            }}
            onAdd={(product, variantId, colorName) => {
              traceEvent("cart-add", { productId: product.id, variantId, count: cart.length + 1 });
              setCart((items) => [...items, { productId: product.id, variantId }]);
              setAdded(`${product.name} (${colorName}) added to cart.`);
            }}
          />
        )}
        {page === "cart" && (
          <CartView
            lines={cartLines}
            subtotal={subtotal}
            quote={quote}
            total={total}
            onPlaceOrder={() => {
              traceEvent("checkout-success", { count: cart.length });
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
              traceEvent("cart-clear", { count: 0 });
              setCart([]);
              setAdded("");
              goToCollection({}, { replace: true });
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          />
        )}
      </main>
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
