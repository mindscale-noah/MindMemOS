import { createHash } from "node:crypto";
import z from "@deepseek-ai/schemastery";
import { createUserMessage } from "@deepseek-ai/dsh-llm";
import type { Context } from "@deepseek-ai/cordis";
import type { Agent } from "@deepseek-ai/dsh-agent";
import type { ContentBlock, UserMessage } from "@deepseek-ai/dsh-llm";
import type { Session, SessionEvent } from "@deepseek-ai/dsh-session";
import { spawnFileJson, spawnFileOk } from "./mindmemos-cli.js";

/**
 * DeepSeek Harness (dsh) plugin that wires MindMemOS into the harness: it
 * `search`es memories before each turn and injects the hits as model context,
 * and `add`s each completed turn through the `mindmemos` CLI.
 *
 * It is a thin layer over the CLI — see the OpenClaw plugin for the same
 * integration on a different host. The CLI must be installed and authenticated
 * separately; this plugin shells out to it for every operation.
 */

type MemoryHit = {
  id?: string;
  memory?: string;
  last_update_at?: string | null;
  event_time?: string | null;
  source_timestamp?: string | null;
};

type MemorySearchResult = {
  request_id?: string | null;
  memories?: MemoryHit[];
};

type MemoryMessage = {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  timestamp: number;
};

type SkillContext = {
  name: string;
  content_hash: string;
  base_version_id: string;
  version_label?: string;
  usage?: "injected" | "modified";
};

const MEMORY_CONTEXT_OPEN = "<relevant-memories>";
const MEMORY_CONTEXT_CLOSE = "</relevant-memories>";

/** Cordis plugin name, also stamped on injected messages for provenance. */
export const name = "mindmemos-memory";

/** The agent registry owns pre-step processing; depend on it so recall runs after agents exist. */
export const inject = ["agents"];

/** Resolved plugin configuration, filled in by the {@link Config} schema before {@link apply} runs. */
export interface Config {
  /** Command used to invoke the CLI (or an absolute path / wrapper command). */
  cli: string;
  /** Number of memories injected per turn. */
  topK: number;
  /** `sync` blocks until extraction finishes; `async` enqueues and returns. */
  addMode: "sync" | "async";
  /** Scopes search and add to one user; omit for project-wide search. */
  userId?: string;
  /** Application scope attached to every search and add. */
  appId: string;
  /** Override the harness session id used as the CLI session scope. */
  sessionId?: string;
  /** Skip recall for prompts shorter than this many characters. */
  minQueryLength: number;
  /** Cap on how many trailing messages are persisted per turn. */
  maxConversationMessages: number;
  /** Minimum supported Python version for the mindmemos CLI (inclusive; the SDK requires >=3.11). */
  minPythonVersion: string;
  /** Maximum supported Python version for the mindmemos CLI (exclusive; the SDK requires <3.14). */
  maxPythonVersion: string;
}

/** Schemastery schema for {@link Config}; cordis validates and defaults the entry config with it. */
export const Config: z<Config> = z.object({
  cli: z.string().default("mindmemos"),
  topK: z.natural().min(1).default(5),
  addMode: z.union([z.const("sync"), z.const("async")]).default("async"),
  userId: z.string(),
  appId: z.string().default("deepseek-harness"),
  sessionId: z.string(),
  minQueryLength: z.natural().min(1).default(2),
  maxConversationMessages: z.natural().min(1).default(80),
  minPythonVersion: z.string().pattern(/^3\.1[1-4]$/).default("3.11"),
  maxPythonVersion: z.string().pattern(/^3\.1[1-4]$/).default("3.14"),
});

export function apply(ctx: Context, config: Config): void {
  // Recall: search and inject memories before the first step of each turn.
  // `agent/pre-step` is a waterfall; prepending lets our injection land ahead of
  // later listeners while preserving whatever messages `next()` already claimed.
  ctx.on(
    "agent/pre-step",
    async ({ agent, messages, step, signal }, next) => {
      const decision = await next();
      if (step !== 1) {
        return decision;
      }
      if (decision.kind === "reject" || signal.aborted) {
        return decision;
      }

      const query = extractQuery(messages);
      if (query.length < config.minQueryLength) {
        return decision;
      }

      try {
        const sessionId = resolveSessionId(config, agentSessionId(agent));
        const result = await searchMemories(config, query, sessionId);
        const context = formatMemoryContext(result.memories ?? [], config.userId);
        if (!context) {
          return decision;
        }

        ctx.logger(name).info(
          `recall hit ${result.memories?.length ?? 0} memories, injected ${context.length} chars`,
        );
        return {
          kind: "enter",
          messages: [
            ...decision.messages,
            createUserMessage({
              content: [{ type: "text", text: context }],
              source: {
                kind: "plugin",
                plugin: name,
                form: "snapshot",
                sections: [{ name, text: context }],
              },
            }),
          ],
        };
      } catch (error) {
        ctx.logger(name).warn(`memory search failed: ${errorMessage(error)}`);
        return decision;
      }
    },
    { prepend: true },
  );

  // Store: on each completed turn, persist that turn's messages. The session log
  // is the source of truth, so no client-side buffer is kept in sync — earlier
  // turns were stored by their own `turn/end`. Injected memory context carries
  // `source.kind === "plugin"` and is excluded to avoid storing our own recall.
  ctx.on("session/event", (session, event) => {
    if (event.type !== "turn/end" || event.data.reason.kind !== "completed") {
      return;
    }
    const messages = collectTurnMessages(session, event.data.turn, config.maxConversationMessages);
    if (messages.length === 0) {
      return;
    }

    const sessionId = resolveSessionId(config, String(session.id));
    void addConversation(config, messages, sessionId)
      .then(() => {
        ctx.logger(name).info(
          `stored ${messages.length} message(s) from turn ${event.data.turn} (session_id=${sessionId})`,
        );
      })
      .catch((error: unknown) => {
        ctx.logger(name).warn(`memory add failed: ${errorMessage(error)}`);
      });
  });

  ctx.logger(name).info("plugin loaded");
}

async function searchMemories(config: Config, query: string, sessionId: string): Promise<MemorySearchResult> {
  const args = ["memory", "search", query, "--top-k", String(config.topK), "--json"];
  args.push("--app-id", config.appId, "--session-id", sessionId);
  if (config.userId) {
    args.push("--user-id", config.userId);
  }
  return spawnFileJson<MemorySearchResult>(config.cli, args);
}

async function addConversation(config: Config, messages: MemoryMessage[], sessionId: string): Promise<void> {
  const args = ["memory", "add", "--messages-json-file", "-", "--json"];
  args.push("--app-id", config.appId, "--session-id", sessionId);
  if (config.addMode === "async") {
    args.push("--async");
  }
  if (config.userId) {
    args.push("--user-id", config.userId);
  }
  const skillContext = detectSkillContext(messages);
  if (skillContext.length > 0) {
    args.push("--skill-context-json", JSON.stringify(skillContext));
  }
  args.push("--metadata-json", JSON.stringify({ source: "deepseek-harness-plugin" }));
  await spawnFileOk(config.cli, args, `${JSON.stringify(messages)}\n`);
}

function formatMemoryContext(memories: MemoryHit[], userId: string | undefined): string {
  const lines = memories
    .map((hit, index) => {
      const text = typeof hit.memory === "string" ? hit.memory.trim() : "";
      if (!text) {
        return null;
      }
      const label = hit.id ? `${index + 1}. [${hit.id}]` : `${index + 1}.`;
      const when = hit.last_update_at ?? hit.event_time ?? hit.source_timestamp;
      return when ? `${label} ${text} (${when})` : `${label} ${text}`;
    })
    .filter((line): line is string => line !== null);

  if (lines.length === 0) {
    return "";
  }
  const preamble = userId ? `Relevant memories for ${userId}:` : "Relevant memories:";
  return [MEMORY_CONTEXT_OPEN, preamble, ...lines, MEMORY_CONTEXT_CLOSE].join("\n");
}

/** The recall query: the real human prompt(s) entering step 1, not injected context. */
function extractQuery(messages: readonly UserMessage[]): string {
  return messages
    .filter((message) => message.source.kind === "user")
    .map((message) => textFromContent(message.content))
    .filter(Boolean)
    .join("\n")
    .trim();
}

/** Collect the surface messages of one completed turn, walking the log back to its `turn/start`. */
function collectTurnMessages(session: Session, turn: number, maxMessages: number): MemoryMessage[] {
  const events = session.events;
  const collected: MemoryMessage[] = [];
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (event.type === "turn/start" && event.data.turn === turn) {
      break;
    }
    const message = messageFromEvent(event);
    if (message !== null) {
      collected.unshift(message);
    }
  }
  return collected.length <= maxMessages ? collected : collected.slice(collected.length - maxMessages);
}

function messageFromEvent(event: SessionEvent): MemoryMessage | null {
  switch (event.type) {
    case "user/message": {
      // Only real human input is stored; plugin-injected context (our own recall,
      // other plugins' snapshots) is excluded to avoid a feedback loop.
      if (event.data.source.kind !== "user") {
        return null;
      }
      const content = textFromContent(event.data.content);
      return content ? { role: "user", content, timestamp: event.time } : null;
    }
    case "assistant/message": {
      const content = textFromContent(event.data.message.content);
      return content ? { role: "assistant", content, timestamp: event.time } : null;
    }
    case "tool/result": {
      const content = textFromContent(event.data.message.content);
      return content ? { role: "tool", content, timestamp: event.time } : null;
    }
    default:
      return null;
  }
}

/** Render message content blocks as one plain-text string; tool calls stay visible as `[tool_call]`. */
function textFromContent(content: readonly ContentBlock[]): string {
  return content
    .map(textFromBlock)
    .filter(Boolean)
    .join("\n")
    .trim();
}

function textFromBlock(block: ContentBlock): string {
  switch (block.type) {
    case "text":
      return block.text;
    case "tool-call":
      return `[tool_call] ${block.name}(${block.arguments})`;
    case "tool-result":
      return textFromContent(block.content);
    default:
      // reasoning and image blocks carry no durable fact text worth storing.
      return "";
  }
}

/** dsh's `Agent.id` is the same `SessionId` as `agent.session.id` ("single identity shared with session"), so recall and store scope identically. */
function agentSessionId(agent: Agent): string {
  return String(agent.id);
}

/** Apply the config override or sanitize the harness session id for use as a CLI argument. */
function resolveSessionId(config: Config, id: string): string {
  return sanitizeSessionId(config.sessionId ?? id);
}

/** Strip control characters (notably NUL) so the value is safe as a CLI argument. */
function sanitizeSessionId(value: string): string {
  // eslint-disable-next-line no-control-regex
  const cleaned = value.replace(/[\u0000-\u001f\u007f]/g, "");
  return cleaned || "deepseek-harness:default";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Detect skill references from the collected turn messages, mirroring the
 * OpenClaw plugin so the same `--skill-context-json` reaches the CLI. dsh's
 * file tools are also named `read`/`write`/`edit`, so the tool-name matching is
 * shared; only `edit` differs — it takes `old_string`/`new_string` rather than a
 * whole `content`/`replacement`, so `new_string` is treated as the new content.
 */
function detectSkillContext(messages: MemoryMessage[]): SkillContext[] {
  const candidates = new Map<string, { path: string; content: string; usage: "injected" | "modified" }>();
  for (let i = 0; i < messages.length; i += 1) {
    const message = messages[i];
    if (message.role !== "assistant") {
      continue;
    }
    const call = parseToolCall(message.content);
    if (!call) {
      continue;
    }
    const path = toolArgPath(call.args);
    if (!path || !isSkillMdPath(path)) {
      continue;
    }
    const key = skillDirKey(path);
    if (call.tool === "read") {
      const content = messages[i + 1]?.role === "tool" ? messages[i + 1].content : "";
      if (content) {
        candidates.set(key, { path, content, usage: strongestUsage(candidates.get(key)?.usage, "injected") });
      }
    } else if (call.tool === "write" || call.tool === "edit") {
      const content = call.tool === "edit" ? editContent(call.args) : toolArgText(call.args, "content");
      if (content) {
        candidates.set(key, { path, content, usage: "modified" });
      }
    }
  }
  return [...candidates.values()].map((candidate) => {
    const metadata = parseSkillMetadata(candidate.content);
    return {
      name: metadata.name || skillNameFromPath(candidate.path),
      content_hash: computeSkillContentHash(candidate.content),
      base_version_id: "",
      ...(metadata.version ? { version_label: metadata.version } : {}),
      usage: candidate.usage,
    };
  });
}

function parseToolCall(content: string): { tool: string; args: Record<string, unknown> } | null {
  const match = content.match(/^\s*\[tool_call\]\s*([A-Za-z0-9_.-]+)\((.*)\)\s*$/s);
  if (!match) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = match[2].trim() ? JSON.parse(match[2]) : {};
  } catch {
    return null;
  }
  if (!isRecord(parsed)) {
    return null;
  }
  return { tool: match[1].toLowerCase(), args: parsed };
}

function toolArgPath(args: Record<string, unknown>): string {
  for (const key of ["path", "file_path", "filepath"]) {
    const value = args[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function toolArgText(args: Record<string, unknown>, key: string): string {
  const value = args[key];
  return typeof value === "string" ? value : "";
}

/** New content of an `edit` call. dsh's `edit` carries `new_string`; OpenClaw-style tools carry a whole content/replacement field. */
function editContent(args: Record<string, unknown>): string {
  for (const key of ["content", "new_content", "replacement", "replace", "new_string"]) {
    const value = toolArgText(args, key);
    if (value) {
      return value;
    }
  }
  return "";
}

function isSkillMdPath(path: unknown): path is string {
  return typeof path === "string" && /(^|[/\\])SKILL\.md$/.test(path);
}

function skillDirKey(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/SKILL\.md$/, "");
}

function skillNameFromPath(path: string): string {
  const parts = skillDirKey(path).split("/").filter(Boolean);
  return parts[parts.length - 1] || "skill";
}

function strongestUsage(
  current: "injected" | "modified" | undefined,
  next: "injected" | "modified",
): "injected" | "modified" {
  return current === "modified" || next === "modified" ? "modified" : "injected";
}

function parseSkillMetadata(content: string): { name?: string; version?: string } {
  return {
    name: simpleFrontmatterField(content, "name"),
    version: simpleFrontmatterField(content, "version"),
  };
}

function simpleFrontmatterField(content: string, field: string): string | undefined {
  const match = content.match(new RegExp(`^\\s*${field}\\s*:\\s*["']?([^"'\\n#]+)`, "m"));
  return match?.[1]?.trim() || undefined;
}

function computeSkillContentHash(content: string): string {
  const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const canonical = JSON.stringify([{ content: normalized, path: "SKILL.md" }]);
  return createHash("sha256").update(canonical).digest("hex");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
