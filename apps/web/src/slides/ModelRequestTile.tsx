import { ArrowRight } from "lucide-react";
import { photo, product, type ModelRequest } from "../lib/model";
import { cn } from "../lib/utils";

/** One evaluation request: the shopper's words, the product that fits, and where it ranked. */
export function ModelRequestTile({ request }: { request: ModelRequest }) {
  const target = product(request.target);
  return (
    <div className="flex items-center gap-3 border-t border-line pt-2">
      <img
        src={photo(request.target)}
        alt={target.name}
        className="size-14 shrink-0 rounded object-cover"
      />
      <div className="min-w-0">
        <p className="line-clamp-2 text-base leading-tight font-medium">{request.request}</p>
        <p className="mt-1 flex items-center gap-1.5 text-sm">
          <span className="text-graphite">#{target.defaultRank}</span>
          <ArrowRight aria-label="to" className="size-3.5" />
          <span className={cn("font-bold", request.modelRank === 1 ? "text-cobalt" : "text-ink")}>
            #{request.modelRank}
          </span>
          <span className="truncate text-muted">{target.name}</span>
        </p>
      </div>
    </div>
  );
}
