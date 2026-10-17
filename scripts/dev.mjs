import net from "node:net";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const services = [
  { name: "api", port: 4311, package: "@storefront/api" },
  { name: "storefront", port: 4312, package: "@storefront/storefront" },
  { name: "dashboard", port: 4310, package: "@storefront/web" },
];
const reservations = [];
try {
  for (const service of services) {
    const server = net.createServer();
    await new Promise((resolve, reject) => {
      server.once("error", () =>
        reject(
          new Error(
            `Port ${service.port} is in use (${service.name}). Stop its owner, then retry pnpm dev. No process was killed.`,
          ),
        ),
      );
      server.listen(service.port, "127.0.0.1", resolve);
    });
    reservations.push(server);
  }
} catch (error) {
  for (const server of reservations) server.close();
  console.error(error.message);
  process.exit(1);
}
await Promise.all(reservations.map((server) => new Promise((resolve) => server.close(resolve))));
const children = [];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const child of children) {
    try {
      if (process.platform === "win32") child.kill("SIGTERM");
      else process.kill(-child.pid, "SIGTERM");
    } catch {
      /* Child already exited. */
    }
  }
  setTimeout(() => process.exit(code), 700).unref();
}
process.once("SIGINT", () => stop());
process.once("SIGTERM", () => stop());
for (const service of services) {
  const child = spawn(
    process.platform === "win32" ? "pnpm.cmd" : "pnpm",
    ["--filter", service.package, "dev"],
    {
      cwd: root,
      stdio: "inherit",
      detached: process.platform !== "win32",
      env: process.env,
    },
  );
  children.push(child);
  child.once("error", (error) => {
    console.error(`${service.name}: ${error.message}`);
    stop(1);
  });
  child.once("exit", (code) => {
    if (!stopping) {
      console.error(`${service.name} exited (${code}); stopping local services.`);
      stop(code || 1);
    }
  });
}
console.log("Storefront Lab — dashboard http://127.0.0.1:4310 · API :4311 · demo shop :4312");
