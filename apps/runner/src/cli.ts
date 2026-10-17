import { parseRunnerInput, protocolFailure } from "./protocol.js";
import { runBrowser } from "./playwright.js";

async function main() {
  // A full run snapshot is tiny. Bound stdin so a malformed caller cannot grow memory indefinitely.
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of process.stdin) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buffer.length;
    if (size > 2 * 1024 * 1024) throw new Error("Runner request exceeds 2 MiB.");
    chunks.push(buffer);
  }
  const input = parseRunnerInput(JSON.parse(Buffer.concat(chunks).toString("utf8")));
  return runBrowser(input);
}

try {
  process.stdout.write(`${JSON.stringify(await main())}\n`);
} catch (error) {
  const output = protocolFailure(error);
  process.stderr.write(`${output.error}\n`);
  process.stdout.write(`${JSON.stringify(output)}\n`);
  process.exitCode = 1;
}
