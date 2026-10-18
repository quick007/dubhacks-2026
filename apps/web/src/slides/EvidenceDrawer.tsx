import { Dialog } from "@base-ui/react/dialog";
import type { Run } from "@storefront/contracts";
import { X } from "lucide-react";
import { RunDetail } from "../components/RunDetail";

const noop = () => {};

export function EvidenceDrawer({
  run,
  variantName,
  onClose,
}: {
  run?: Run;
  variantName: string;
  onClose: () => void;
}) {
  return (
    <Dialog.Root open={run !== undefined} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-20 bg-ink/30" />
        <Dialog.Popup className="fixed inset-y-0 right-0 z-30 w-full max-w-md overflow-y-auto bg-fog p-4 shadow-xl">
          <div className="mb-3 flex items-center justify-between">
            <Dialog.Title className="font-display text-3xl font-extrabold tracking-[-0.015em]">
              Evidence
            </Dialog.Title>
            <Dialog.Close
              aria-label="Close"
              className="rounded-md p-1.5 text-muted hover:bg-paper hover:text-ink"
            >
              <X className="size-5" />
            </Dialog.Close>
          </div>
          {run && <RunDetail run={run} variantName={variantName} history={[]} onSelectRun={noop} />}
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
