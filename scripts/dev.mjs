import net from "node:net";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { LOCAL_ORIGINS, LOCAL_PROXY_PORT } from "../packages/contracts/src/local.ts";

const root = fileURLToPath(new URL("../", import.meta.url));
const cli = path.join(root, "node_modules/portless/dist/cli.js");
// Never inherit a privileged, TLS, hosts-file or network-sharing configuration.
const env = {
  ...process.env,
  PORTLESS: "1",
  PORTLESS_PORT: String(LOCAL_PROXY_PORT),
  PORTLESS_HTTPS: "0",
  PORTLESS_LAN: "0",
  PORTLESS_SYNC_HOSTS: "0",
  PORTLESS_TLD: "localhost",
  PORTLESS_WILDCARD: "0",
  PORTLESS_TAILSCALE: "0",
  PORTLESS_FUNNEL: "0",
  PORTLESS_NGROK: "0",
  PORTLESS_STATE_DIR: path.join(root, ".cache/portless"),
};
delete env.PORTLESS_APP_PORT;
delete env.PORTLESS_LAN_IP;
delete env.PORT;

// Refuse an unrelated listener; never take a route/port over with --force.
const reservation = net.createServer();
try {
  await new Promise((resolve, reject) => {
    reservation.once("error", reject);
    reservation.listen(LOCAL_PROXY_PORT, "127.0.0.1", resolve);
  });
  await new Promise((resolve) => reservation.close(resolve));
} catch {
  console.error(
    `Portless HTTP proxy port ${LOCAL_PROXY_PORT} is in use. Stop this project's previous pnpm dev, then retry. No process was killed.`,
  );
  process.exit(1);
}

const children = [];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  process.exitCode = code;
  for (const child of children) {
    try {
      if (process.platform === "win32") child.kill("SIGTERM");
      else process.kill(-child.pid, "SIGTERM");
    } catch {
      /* Already stopped. */
    }
  }
  setTimeout(() => {
    for (const child of children) {
      if (child.exitCode === null && child.signalCode === null) {
        try {
          if (process.platform !== "win32") process.kill(-child.pid, "SIGKILL");
        } catch {
          /* Already stopped. */
        }
      }
    }
    process.exit(code);
  }, 5000).unref();
}
function launch(args) {
  const child = spawn(process.execPath, [cli, ...args], {
    cwd: root,
    env,
    stdio: "inherit",
    detached: process.platform !== "win32",
  });
  children.push(child);
  child.once("error", (error) => {
    console.error(error.message);
    stop(1);
  });
  child.once("exit", (code) => {
    if (!stopping) stop(code || 1);
  });
  return child;
}
process.once("SIGINT", () => stop());
process.once("SIGTERM", () => stop());
launch(["proxy", "start", "--foreground", "--no-tls", "--port", String(LOCAL_PROXY_PORT)]);
try {
  const deadline = Date.now() + 10_000;
  let ready = false;
  while (!stopping && Date.now() < deadline) {
    ready = await new Promise((resolve) => {
      const socket = net.connect(LOCAL_PROXY_PORT, "127.0.0.1");
      socket.once("connect", () => {
        socket.destroy();
        resolve(true);
      });
      socket.once("error", () => {
        socket.destroy();
        resolve(false);
      });
    });
    if (ready) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  if (!ready || stopping) throw new Error("Project Portless proxy did not start.");
  launch([]); // Official workspace discovery and named routes from portless.json.
  console.log(
    `Storefront Lab — ${LOCAL_ORIGINS.dashboard} · API ${LOCAL_ORIGINS.api} · shop ${LOCAL_ORIGINS.storefront}`,
  );
} catch (error) {
  console.error(error.message);
  stop(1);
}
