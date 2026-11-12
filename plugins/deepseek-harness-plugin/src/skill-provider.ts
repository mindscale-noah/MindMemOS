import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Context } from "@deepseek-ai/cordis";
import { BUNDLED_SKILL_RANK, renderSkillContent, type SkillDefinition } from "@deepseek-ai/dsh-skill";

/** Recover the exact source of a native load whose model-facing body we observed. */
export async function nativeSkillRead(ctx: Context, rendered: string, cwd: string | undefined) {
  const name = rendered.match(/^<skill_content name="([a-z0-9-]+)">/)?.[1];
  if (!name) {
    return undefined;
  }
  try {
    const skill = await ctx.skills.get(name, { cwd, signal: AbortSignal.timeout(1_000) });
    if (!skill?.path || basename(skill.path) !== "SKILL.md" || renderSkillContent(skill) !== rendered.trim()) {
      return undefined;
    }
    const content = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(await readFile(skill.path));
    const body = content.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n([\s\S]*)$/)?.[1];
    if (body?.trim() === skill.content.trim()) {
      return { kind: "read" as const, path: skill.path, content };
    }
  } catch {
    // A provider may no longer have the observed local source.
  }
  return undefined;
}

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
