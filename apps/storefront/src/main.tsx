import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import { orderedProducts, parseVariantConfig, PRODUCTS, type Product } from "./config";
import "./styles.css";

const config = parseVariantConfig(window.location.search);
function ProductArt({ product }: { product: Product }) {
  return (
    <div className={`product-art ${product.color}`} aria-hidden="true">
      <div className={`object ${product.kind}`}>
        <span />
      </div>
      <span className="art-shadow" />
    </div>
  );
}
function App() {
  const [cart, setCart] = useState<string[]>([]);
  const [page, setPage] = useState<"shop" | "cart" | "confirmed">("shop");
  const [added, setAdded] = useState("");
  const total = cart.reduce(
    (sum, id) => sum + (PRODUCTS.find((product) => product.id === id)?.price ?? 0),
    0,
  );
  return (
    <>
      <div className="demo-banner">LOCAL DEMO STORE · No real payments, purchases or shipping</div>
      <header>
        <a className="wordmark" href="/">
          fieldwork<span>®</span>
        </a>
        <span className="header-note">Useful things. Thoughtfully simple.</span>
        <button aria-label="View cart" className="cart-button" onClick={() => setPage("cart")}>
          View cart<span className="cart-count">{cart.length}</span>
        </button>
      </header>
      <main>
        {page === "shop" && (
          <>
            <section className="hero">
              <p className="eyebrow">A small collection, made for everyday.</p>
              <h1>{config.headline}</h1>
              <p>
                Carry less. Take the long way home. A few useful companions for the moments in
                between.
              </p>
              {config.showShipping && (
                <div className="shipping-info">
                  Free demo shipping on orders over $50 · $5 flat rate otherwise
                </div>
              )}
              <a href="#collection" className="collection-link">
                Meet the collection <span>↓</span>
              </a>
            </section>
            <section id="collection" className="collection">
              <div className="section-heading">
                <h2>Everyday companions</h2>
                <p>
                  {config.recommendationOrder === "budget"
                    ? "Small favorites, lowest price first."
                    : config.recommendationOrder === "history"
                      ? "Demo history ordering · no personal history collected."
                      : "A few of our favorites."}
                </p>
              </div>
              <div className="products">
                {orderedProducts(config.recommendationOrder).map((product) => (
                  <article className="product" key={product.id}>
                    <ProductArt product={product} />
                    <div className="product-heading">
                      <h3>{product.name}</h3>
                      <span>${product.price}</span>
                    </div>
                    <p>{product.category}</p>
                    <button
                      className="add-button"
                      onClick={() => {
                        setCart((items) => [...items, product.id]);
                        setAdded(`${product.name} added to cart.`);
                      }}
                    >
                      Add to cart
                    </button>
                  </article>
                ))}
              </div>
              <p className="added-message" role="status">
                {added}
              </p>
            </section>
          </>
        )}
        {page === "cart" && (
          <section className="cart-page">
            <p className="eyebrow">A little adventure, packed.</p>
            <h1>Your demo cart</h1>
            {cart.length === 0 ? (
              <p>Your cart is empty. Find an everyday companion in the collection.</p>
            ) : (
              <>
                <div className="cart-items">
                  {PRODUCTS.filter((product) => cart.includes(product.id)).map((product) => (
                    <div key={product.id}>
                      <span>
                        {product.name}{" "}
                        <small>× {cart.filter((id) => id === product.id).length}</small>
                      </span>
                      <span>${product.price * cart.filter((id) => id === product.id).length}</span>
                    </div>
                  ))}
                </div>
                <div className="cart-total">
                  <span>Demo subtotal</span>
                  <strong>${total}</strong>
                </div>
                <p className="demo-note">
                  This is a fake order. No payment, address or email is required. Nothing will be
                  purchased or shipped.
                </p>
                <button className="primary-button" onClick={() => setPage("confirmed")}>
                  Place demo order
                </button>
              </>
            )}
            <button className="text-button" onClick={() => setPage("shop")}>
              Continue browsing
            </button>
          </section>
        )}
        {page === "confirmed" && (
          <section className="confirmation">
            <span className="checkmark" aria-hidden="true">
              ✓
            </span>
            <p className="eyebrow">The fake checkout worked.</p>
            <h1>Demo order confirmed</h1>
            <p>
              Your local test reached its goal. No payment was processed and no real order was
              placed.
            </p>
            <div className="receipt">
              {cart.length} demo item{cart.length === 1 ? "" : "s"} · ${total} demo subtotal
            </div>
            <button
              className="primary-button"
              onClick={() => {
                setPage("shop");
                setCart([]);
                setAdded("");
              }}
            >
              Back to the collection
            </button>
          </section>
        )}
      </main>
      <footer>
        <span className="wordmark">
          fieldwork<span>®</span>
        </span>
        <p>Built for the everyday. Here for the demo.</p>
        <span>Local test shop / synthetic shoppers</span>
      </footer>
    </>
  );
}
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
