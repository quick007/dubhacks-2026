import path from "node:path";
import { RunSchema, type Artifact, type Run, type StepEvent } from "@storefront/contracts";

/** One request per child process; the Go API owns IDs, persistence and evaluation. */
export interface RunnerInput {
  run: Run;
  artifactDirectory: string;
}
export interface RunnerEvent {
  kind: StepEvent["kind"];
  message: string;
}
export interface RunnerArtifact {
  kind: Artifact["kind"];
  fileName: string;
  mimeType: string;
}
export interface RunnerResult {
  taskCompleted: boolean;
  failureReasons: string[];
  notes: string[];
}
export interface RunnerOutput {
  events: RunnerEvent[];
  artifacts: RunnerArtifact[];
  result: RunnerResult;
  /** Fatal input/protocol error. CLI exits 1; a browser task failure instead exits 0. */
  error?: string;
}

export function parseRunnerInput(input: unknown): RunnerInput {
  if (!input || typeof input !== "object" || Array.isArray(input))
    throw new Error("Expected one runner request object.");
  const request = input as Record<string, unknown>;
  if (Object.keys(request).some((key) => key !== "run" && key !== "artifactDirectory"))
    throw new Error("Runner request accepts only run and artifactDirectory.");
  if (
    typeof request.artifactDirectory !== "string" ||
    !path.isAbsolute(request.artifactDirectory) ||
    request.artifactDirectory.includes("\0")
  )
    throw new Error("artifactDirectory must be an absolute local directory path.");
  const artifactDirectory = path.resolve(request.artifactDirectory);
  if (artifactDirectory === path.parse(artifactDirectory).root)
    throw new Error("artifactDirectory cannot be the filesystem root.");
  const run = RunSchema.parse(request.run);
  if (run.adapter !== "playwright")
    throw new Error("The browser runner accepts only playwright runs.");
  return { run, artifactDirectory };
}

export function protocolFailure(error: unknown): RunnerOutput {
  const message = error instanceof Error ? error.message : String(error);
  return {
    events: [{ kind: "failure", message }],
    artifacts: [],
    result: {
      taskCompleted: false,
      failureReasons: [message],
      notes: ["Local scripted browser task; synthetic, not human conversion evidence."],
    },
    error: message,
  };
}
