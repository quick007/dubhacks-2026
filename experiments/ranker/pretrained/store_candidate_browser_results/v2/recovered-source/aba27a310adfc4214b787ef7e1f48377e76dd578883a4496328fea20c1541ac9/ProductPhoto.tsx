import type { ProductImage } from "./images";
import { responsiveProductImage } from "./product-responsive-source";

export function ProductPhoto({
  image,
  id,
  alt = image.alt,
  thumbnail = false,
}: {
  image: ProductImage;
  id?: string;
  alt?: string;
  thumbnail?: boolean;
}) {
  const tile = image.tile;
  const width = tile ? Math.max(tile.columns, tile.rows * tile.sheetAspect) * 100 : 100;
  const height = tile ? Math.max(tile.rows, tile.columns / tile.sheetAspect) * 100 : 100;
  const delivery = responsiveProductImage(image.src, tile ? width / 100 : undefined, thumbnail);
  return (
    <span className="product-photo">
      <img
        id={id}
        src={delivery.src}
        srcSet={delivery.srcSet}
        sizes={delivery.sizes}
        onError={(event) => {
          event.currentTarget.removeAttribute("srcset");
          if (event.currentTarget.getAttribute("src") !== image.src) event.currentTarget.src = image.src;
        }}
        alt={alt}
        width={724}
        height={724}
        loading="lazy"
        style={
          tile
            ? {
                position: "absolute",
                width: `${width}%`,
                height: `${height}%`,
                left: `${-(tile.index % tile.columns) * (width / tile.columns) - (width / tile.columns - 100) / 2}%`,
                top: `${-Math.floor(tile.index / tile.columns) * (height / tile.rows) - (height / tile.rows - 100) / 2}%`,
              }
            : { objectPosition: image.position }
        }
      />
    </span>
  );
}
