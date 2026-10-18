import React, { lazy, Suspense, useEffect, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { HugeiconsIcon } from "@hugeicons/react";
import { ShoppingCart01Icon } from "@hugeicons/core-free-icons";
import {
  parseVariantConfig,
  orderedProducts,
  type Product,
} from "./config";
import { getCartLines, type CartItem } from "./cart";
import { personalizeStorefront } from "./personalization";
import { COLLECTIONS } from "./collections";
import { orderTotal, shippingQuote } from "./shipping";
import { INITIAL_PERSONA_ID, SHOPPER_PERSONAS, type ShopperPersona } from "./personas";
import { checkoutStillCurrent, runCurrentCheckout, switchShoppingPerson } from "./shopping-session";
import { ShopView } from "./ShopView";
import { CartView } from "./CartView";
import { ConfirmationView } from "./ConfirmationView";
import { useRoute } from "./use-route";
import type { StorefrontRoute } from "./routes";
import { localModelEnabled, localModelEndpoint } from "./local-model";
import { useLocalModel } from "./use-local-model";
import { demoTraceEnabled, traceContext, traceEvent, traceProducts } from "./session-trace-bridge";
import { confirmDemoCheckout, type DemoCheckoutReceipt } from "./demo-checkout-client";
import "./styles.css";

const TraceControls = lazy(() => import("./TraceControls"));
const PersonaSwitcher = lazy(() => import("./PersonaSwitcher").then(module => ({ default: module.PersonaSwitcher })));

const initialConfig = parseVariantConfig(window.location.search);

function App() {
  const { route, navigate } = useRoute();
  const config = initialConfig;
  const [personId, setPersonId] = useState(INITIAL_PERSONA_ID);
  const person = SHOPPER_PERSONAS.find(item => item.id === personId)!;
  const carts = useRef<Record<string, CartItem[]>>({});
  const [cart, setCart] = useState<CartItem[]>([]);
  const selectionScope = `${personId}:${route.query}`;
  const [selection, setSelection] = useState<{ scope: string; values: Partial<Record<Product["id"], string>> }>({ scope: "", values: {} });
  const selectedVariants = selection.scope === selectionScope ? selection.values : {};
  const [added, setAdded] = useState("");
  const [searchInput, setSearchInput] = useState(route.query);
  const [checkoutPending, setCheckoutPending] = useState(false);
  const [checkoutError, setCheckoutError] = useState("");
  const [checkoutReceipt, setCheckoutReceipt] = useState<DemoCheckoutReceipt | null>(null);
  const checkoutId = useRef(crypto.randomUUID());
  const checkoutInFlight = useRef<string | null>(null);

  useEffect(() => {
    setSearchInput(route.query);
  }, [route.query]);

  const personalizedDefault = import.meta.env.VITE_PERSONALIZED_STORE === "true";
  const localMode = localModelEnabled(window.location.search, personalizedDefault);
  const localModel = useLocalModel(localMode && route.page === "shop", route.query, person.request,
    localModelEndpoint(window.location.search, personalizedDefault));
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
  // A restored link needs both cart contents and a server-confirmed receipt.
  const page = route.page === "confirmed" && (cart.length === 0 || !checkoutReceipt) ? "cart" : route.page;

  useEffect(() => {
    document.title =
      page === "cart"
        ? "Fieldwork — Your cart"
        : page === "confirmed"
          ? "Fieldwork — Order confirmation"
          : route.query
            ? `Fieldwork — ${route.query}`
            : "Fieldwork — Everyday essentials";
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

  function resetCheckout() {
    checkoutId.current = crypto.randomUUID();
    checkoutInFlight.current = null;
    setCheckoutPending(false);
    setCheckoutReceipt(null);
    setCheckoutError("");
  }

  async function placeDemoOrder() {
    if (checkoutInFlight.current) return;
    const id = checkoutId.current;
    checkoutInFlight.current = id;
    setCheckoutPending(true); setCheckoutError("");
    await runCurrentCheckout(() => confirmDemoCheckout(cart, config.showShipping, id),
      () => checkoutStillCurrent(id, checkoutId.current), {
        confirmed: receipt => {
          setCheckoutReceipt(receipt);
          traceEvent("checkout-success", { count: cart.length });
          navigate({ ...route, page: "confirmed" });
          window.scrollTo({ top: 0, behavior: "smooth" });
        },
        failed: () => setCheckoutError("Your order could not be confirmed. Please try again."),
        settled: () => {
          if (checkoutInFlight.current === id) { checkoutInFlight.current = null; setCheckoutPending(false); }
        },
      });
  }

  return (
    <>
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
      <main data-demo-checkout-id={checkoutReceipt?.checkoutId} data-demo-receipt-id={checkoutReceipt?.receiptId}>
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
            personName={localMode ? person.displayName : undefined}
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
              setSelection(previous => ({ scope: selectionScope, values: { ...(previous.scope === selectionScope ? previous.values : {}), [productId]: variantId } }));
            }}
            onAdd={(product, variantId, colorName) => {
              resetCheckout();
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
            checkoutPending={checkoutPending}
            checkoutError={checkoutError}
            onPlaceOrder={() => { void placeDemoOrder(); }}
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
              resetCheckout();
              traceEvent("cart-clear", { count: 0 });
              setCart([]);
              setAdded("");
              goToCollection({}, { replace: true });
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          />
        )}
      </main>
      {localMode && <Suspense fallback={null}><PersonaSwitcher people={SHOPPER_PERSONAS} person={person} onChange={(next: ShopperPersona) => {
        if (next.id === personId) return;
        const switched = switchShoppingPerson({ personId, cart, carts: carts.current, checkoutReceipt }, next.id);
        carts.current = switched.carts;
        setCart(switched.cart);
        setPersonId(switched.personId);
        setSelection({ scope: "", values: {} });
        setAdded("");
        resetCheckout();
        if (route.page !== "shop") goToCollection({});
      }} /></Suspense>}
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
