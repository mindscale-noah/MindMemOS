import { copyFileSync, cpSync, mkdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";

const generated = resolve("src/generated");
mkdirSync(generated, { recursive: true });
copyFileSync(new URL("./mindmemos-cli.ts", import.meta.url), resolve(generated, "mindmemos-cli.ts"));

if (process.argv.includes("--build")) {
  rmSync("dist", { recursive: true, force: true });
  const skill = new URL("../../skills/mindmemos-cli/", import.meta.url);
  cpSync(skill, "dist/skills/mindmemos-cli", { recursive: true });
}
