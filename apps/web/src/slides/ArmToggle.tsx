import { Toggle } from "@base-ui/react/toggle";
import { ToggleGroup } from "@base-ui/react/toggle-group";
import type { Arm } from "../lib/story";

export const armLabels: Record<Arm, string> = {
  baseline: "Today's shop",
  candidate: "With the fix",
};

export function ArmToggle({ value, onChange }: { value: Arm; onChange: (arm: Arm) => void }) {
  return (
    <ToggleGroup
      value={[value]}
      onValueChange={(next) => next[0] && onChange(next[0] as Arm)}
      className="inline-flex rounded-md bg-paper p-1"
    >
      {(["baseline", "candidate"] as const).map((option) => (
        <Toggle
          key={option}
          value={option}
          className="h-9 rounded px-3 text-muted data-[pressed]:bg-ink data-[pressed]:text-paper"
        >
          {armLabels[option]}
        </Toggle>
      ))}
    </ToggleGroup>
  );
}
