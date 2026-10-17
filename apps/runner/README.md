# Local browser runner

This package owns only the scripted Playwright browser adapter. The Go API owns
the queue, deterministic mock adapter, metadata, evaluation and artifact IDs.

Run one request per child process, from any working directory:

```sh
node --import /absolute/workspace/node_modules/tsx/dist/loader.mjs /absolute/workspace/apps/runner/src/cli.ts
```

Write one JSON object to stdin, then close stdin:

```ts
{ run: Run, artifactDirectory: '/absolute/path/to/run-artifacts' }
```

`Run` is validated against the frozen shared `RunSchema`; `adapter` must be
`playwright`. The input directory is created if needed. Only the fixed filenames
`screenshot.png`, `trace.zip`, and `result.json` are written there.

Stdout contains exactly one JSON object followed by a newline:

```ts
{
  events: Array<{ kind: StepEvent['kind']; message: string }>,
  artifacts: Array<{ kind: Artifact['kind']; fileName: string; mimeType: string }>,
  result: { taskCompleted: boolean; failureReasons: string[]; notes: string[] },
  error?: string
}
```

The Go caller adds persistent event/artifact IDs, timestamps and run IDs.
Normal browser task failures exit 0 with `result.taskCompleted: false` and a
failure event. Invalid input or an unhandled infrastructure error exits 1,
preserving the output shape and adding `error`. Read stdout even on nonzero exit.
Diagnostics go to stderr. Each request is limited to 2 MiB.

The browser permits only the local demo origin `http://shop.dubhacks.localhost:1356` (including
its named Vite HMR websocket through Portless), blocks service workers, applies allowlisted variant
settings, and uses fixed roles through the fake order-confirmation screen.
All evaluation remains synthetic; this does not measure human conversion lift.

Protocol types live in `src/protocol.ts`. Add browser behavior in
`src/playwright.ts`; do not add orchestration or model code here.
