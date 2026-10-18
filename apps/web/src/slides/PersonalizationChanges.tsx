import { BadgeCheck, MessageSquareText, Tags } from "lucide-react";

const changes = [
  {
    icon: BadgeCheck,
    title: "A pick with his name on it",
    detail: 'The "Picked for Bob" badge makes the recommended throw easy to spot.',
  },
  {
    icon: Tags,
    title: "Useful details, at a glance",
    detail: 'Cotton, throw blanket, and "For your sofa" chips put the relevant facts on the card.',
  },
  {
    icon: MessageSquareText,
    title: "A reason to choose it",
    detail: '"Why this fits" connects the product to his request before he opens it.',
  },
];

export function PersonalizationChanges() {
  return (
    <aside
      aria-label="What changed for Bob"
      className="flex flex-col justify-center gap-6 lg:gap-8"
    >
      <h3 className="text-2xl font-bold tracking-tight">What changed for Bob</h3>
      <ul className="grid gap-5 sm:grid-cols-3 lg:grid-cols-1 lg:gap-7">
        {changes.map(({ icon: Icon, title, detail }) => (
          <li key={title} className="flex gap-3">
            <Icon aria-hidden className="mt-0.5 size-5 shrink-0 text-cobalt" />
            <div>
              <p className="text-base leading-snug font-bold">{title}</p>
              <p className="mt-1.5 text-sm leading-relaxed text-muted">{detail}</p>
            </div>
          </li>
        ))}
      </ul>
      <p className="border-t border-line pt-4 text-xs leading-relaxed text-muted">
        Designed demo treatment. Product order comes from the measured model run; badges and details
        use the shopper's request and catalog facts.
      </p>
    </aside>
  );
}
