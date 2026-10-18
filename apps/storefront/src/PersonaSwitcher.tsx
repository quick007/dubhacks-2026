import { useRef } from "react";
import { Popover } from "@base-ui/react/popover";
import { Button } from "@base-ui/react/button";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowUpDownIcon, Cancel01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { personInitials, recentHistory, type ShopperPersona } from "./personas";

export function PersonaSwitcher({ people, person, onChange }: {
  people: ShopperPersona[];
  person: ShopperPersona;
  onChange: (person: ShopperPersona) => void;
}) {
  const selectedOption = useRef<HTMLElement>(null);
  return (
    <aside className="person-switcher" aria-label="Shopping profile">
      <Popover.Root>
      <Popover.Trigger className="person-trigger"
        aria-label={`Shopping as ${person.displayName}. Switch person`}>
        <span className="person-avatar" aria-hidden="true">{personInitials(person.displayName)}</span>
        <span><small>Shopping as</small><strong>{person.displayName}</strong></span>
        <HugeiconsIcon icon={ArrowUpDownIcon} size={17} aria-hidden="true" />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Positioner className="person-positioner" side="top" align="end" sideOffset={12} collisionPadding={16} positionMethod="fixed">
      <Popover.Popup className="person-popover" initialFocus={selectedOption} finalFocus>
        <div className="person-panel-heading">
          <Popover.Title render={<h2 />}>Who’s shopping?</Popover.Title>
          <Popover.Close className="person-close" aria-label="Close shopping profiles">
            <HugeiconsIcon icon={Cancel01Icon} size={18} aria-hidden="true" />
          </Popover.Close>
        </div>
        <div className="person-options" role="group" aria-label="People">
          {people.map(option => (
            <Button key={option.id} ref={option.id === person.id ? selectedOption : undefined} aria-pressed={option.id === person.id}
              onClick={() => onChange(option)}>
              <span className="person-avatar" aria-hidden="true">{personInitials(option.displayName)}</span>
              <span><strong>{option.displayName}</strong><small>{option.role}</small></span>
              {option.id === person.id && <HugeiconsIcon icon={Tick02Icon} size={16} aria-hidden="true" />}
            </Button>
          ))}
        </div>
        <div className="person-background">
          <h3>{person.displayName}’s day to day</h3>
          <p>{person.summary}</p>
          {person.request.persona?.colorPreferences.length ? <p className="person-preferences">
            Favorite colors: {person.request.persona.colorPreferences.join(", ")}
          </p> : null}
          <h3>Recently</h3>
          <ul className="person-history">
            {recentHistory(person).map(event => (
              <li key={event.eventId}>
                <span><small>{event.eventType === "view" ? "Viewed" : event.eventType === "cart" ? "Added to cart" : "Purchased"}</small>
                  {event.itemName}{event.selectedColor ? ` · ${event.selectedColor}` : ""}</span>
                <time dateTime={event.occurredAt}>{new Date(event.occurredAt).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" })}</time>
              </li>
            ))}
          </ul>
        </div>
      </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
      </Popover.Root>
    </aside>
  );
}
