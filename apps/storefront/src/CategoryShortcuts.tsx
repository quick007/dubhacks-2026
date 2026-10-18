import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { COLLECTIONS } from "./collections";
import { PRODUCT_IMAGES } from "./images";
import { ProductPhoto } from "./ProductPhoto";

export function CategoryShortcuts({ collectionId, onChange }: {
  collectionId: string;
  onChange: (id: string) => void;
}) {
  return (
    <nav className="category-shortcuts" aria-label="Shop by category">
      <div className="category-photo-row">
        {COLLECTIONS.filter((collection) => collection.id !== "all").map((collection) => {
          const images = PRODUCT_IMAGES[collection.productIds[0]];
          const image = images.find((view) => view.label === "Still life") ?? images.find((view) => view.presentation.subject === "in-use") ?? images[0];
          return (
            <button type="button" key={collection.id} aria-pressed={collectionId === collection.id} onClick={() => onChange(collection.id)}>
              <span className="category-photo"><ProductPhoto image={image} alt="" /></span>
              <span className="category-name">{collection.label}<HugeiconsIcon icon={ArrowRight01Icon} size={18} aria-hidden="true" /></span>
            </button>
          );
        })}
      </div>
      <button className="all-products-link" type="button" aria-pressed={collectionId === "all"} onClick={() => onChange("all")}>View all products<HugeiconsIcon icon={ArrowRight01Icon} size={18} aria-hidden="true" /></button>
    </nav>
  );
}
