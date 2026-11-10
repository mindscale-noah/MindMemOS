import { readFileSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Context } from "@deepseek-ai/cordis";
import { BUNDLED_SKILL_RANK, type SkillDefinition } from "@deepseek-ai/dsh-skill";

export function registerBundledSkill(ctx: Context): void {
  const path = fileURLToPath(new URL("./skills/mindmemos-cli/SKILL.md", import.meta.url));
  const raw = readFileSync(path, "utf-8");
  const document = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/);
  const name = document?.[1].match(/^name: (.+)$/m)?.[1].trim();
  const description = document?.[1].match(/^description: (.+)$/m)?.[1].trim();
  if (!document || !name || !description) {
    throw new Error("Invalid bundled MindMemOS SKILL.md");
  }

  const definition: SkillDefinition = {
    name,
    description,
    content: document[2].trim(),
    source: "bundled",
    provider: "mindmemos-cli",
    invocation: { modelInvocable: true, userInvocable: true },
    path,
    resourceBase: { kind: "directory", path: dirname(path) },
  };
  const { content, ...summary } = definition;
  ctx.skills.registerProvider(() => ({
    name: definition.provider,
    list: async () => [{ ...summary, locator: name, rank: BUNDLED_SKILL_RANK }],
    get: async (candidate) => candidate.locator === name ? { ...summary, content } : undefined,
  }));
}
