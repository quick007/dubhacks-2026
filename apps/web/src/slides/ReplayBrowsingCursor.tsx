import { Bot, MousePointer2 } from "lucide-react";
import { useEffect, useRef } from "react";
import { REPLAY_STEP_MS } from "../lib/replay";

export function ReplayBrowsingCursor({
  motion,
  name,
  playing,
}: {
  motion: Keyframe[];
  name?: string;
  playing: boolean;
}) {
  const element = useRef<HTMLSpanElement>(null);
  const animation = useRef<Animation | undefined>(undefined);
  useEffect(() => {
    const next = element.current!.animate(motion, { duration: REPLAY_STEP_MS, fill: "both" });
    animation.current = next;
    return () => next.cancel();
  }, [motion]);
  useEffect(() => {
    if (playing) animation.current?.play();
    else animation.current?.pause();
  }, [motion, playing]);

  return (
    <span
      ref={element}
      aria-hidden
      className="replay-cursor pointer-events-none absolute inset-0"
      style={{ transform: motion[0].transform as string }}
    >
      <MousePointer2
        className="absolute -top-1 -left-1 size-8 fill-cobalt text-paper drop-shadow-md"
        strokeWidth={2}
      />
      {name && (
        <span className="absolute top-7 left-3 flex -translate-x-1/2 items-center gap-1 rounded-full bg-cobalt px-2 py-0.5 text-sm font-bold whitespace-nowrap text-paper shadow-md ring-2 ring-paper">
          <Bot className="size-3.5" />
          {name}
        </span>
      )}
    </span>
  );
}
