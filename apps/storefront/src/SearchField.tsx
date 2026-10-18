import { useState } from "react";
import { Combobox } from "@base-ui/react/combobox";
import { HugeiconsIcon } from "@hugeicons/react";
import { Search01Icon } from "@hugeicons/core-free-icons";

export function SearchField({ value, suggestions, onChange, onSuggestion, onFocus }: {
  value: string;
  suggestions: readonly string[];
  onChange: (value: string) => void;
  onSuggestion: (query: string) => void;
  onFocus?: () => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <Combobox.Root items={[...suggestions]} filter={null} inputValue={value}
      onInputValueChange={(next, details) => {
        if (["input-change", "input-clear", "clear-press"].includes(details.reason)) onChange(next);
      }}
      open={open && suggestions.length > 0} onOpenChange={setOpen}
      onValueChange={selected => { if (typeof selected === "string" && selected) { setOpen(false); onSuggestion(selected); } }}>
      <div className="search-field">
        <HugeiconsIcon icon={Search01Icon} size={22} aria-hidden="true" />
        <Combobox.Input id="shop-search" type="search" placeholder="Search the collection" maxLength={200}
          onFocus={() => { setOpen(true); onFocus?.(); }} />
        <button type="submit">Search</button>
      </div>
      <Combobox.Portal>
        <Combobox.Positioner className="search-suggestions-positioner" sideOffset={8} align="start">
          <Combobox.Popup className="search-suggestions">
            <p>Suggested searches</p>
            <Combobox.List>{(query: string) => <Combobox.Item key={query} value={query}>{query}</Combobox.Item>}</Combobox.List>
          </Combobox.Popup>
        </Combobox.Positioner>
      </Combobox.Portal>
    </Combobox.Root>
  );
}
