import { createHash, randomUUID } from "node:crypto";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { buildJsonPluginConfigSchema, definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import {
  asRecord, loadedSkillVersions, reportTrajectory, skillEvidenceText, spawnFileJson,
  type ToolCall, type TrajectoryMessage,
} from "./generated/mindmemos-cli.js";

type PluginConfig = {
  enabled: boolean;
  cli: string;
  topK: number;
  tokenBudget?: number;
  /** Search stored images too; image hits carry a short-lived directly-fetchable image_url. */
  multimodal: boolean;
  addMode: "sync" | "async";
  userId?: string;
  appId: string;
  sessionId?: string;
  minQueryLength: number;
  maxConversationMessages: number;
};

type MemoryHit = {
  id?: string;
  memory?: string;
  last_update_at?: string | null;
  event_time?: string | null;
  source_timestamp?: string | null;
  metadata?: {
    image_url?: string;
    [key: string]: unknown;
  };
};

type MemorySearchResult = {
  request_id?: string | null;
  memories?: MemoryHit[];
};

const MEMORY_CONTEXT_OPEN = "<relevant-memories>";
const MEMORY_CONTEXT_CLOSE = "</relevant-memories>";

const DEFAULT_CONFIG: PluginConfig = {
  enabled: true,
  cli: "mindmemos",
  topK: 5,
  multimodal: true,
  // sync by default: async enqueues through Kafka and silently stores nothing on a
  // server without kafka.enabled=true — sync works on every deployment.
  addMode: "sync",
  appId: "openclaw",
  minQueryLength: 2,
  maxConversationMessages: 80,
};

const CONFIG_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    enabled: {
      type: "boolean",
    },
    cli: {
      type: "string",
    },
    topK: {
      type: "integer",
      minimum: 1,
    },
    multimodal: {
      type: "boolean",
      description: "Search stored images too (image hits carry a short-lived fetchable image_url).",
    },
    tokenBudget: {
      type: "integer",
      minimum: 1,
    },
    addMode: {
      type: "string",
      enum: ["sync", "async"],
    },
    userId: {
      type: "string",
    },
    appId: {
      type: "string",
    },
    sessionId: {
      type: "string",
    },
    minQueryLength: {
      type: "integer",
      minimum: 1,
    },
    maxConversationMessages: {
      type: "integer",
      minimum: 1,
    },
  },
} as const;

const entry: ReturnType<typeof definePluginEntry> = definePluginEntry({
  id: "mindmemos-memory",
  name: "MindMemOS Memory",
  description:
    "Recalls MindMemOS memories and reports completed OpenClaw turns to Memory and Skill.",
  configSchema: buildJsonPluginConfigSchema(CONFIG_SCHEMA),
  register,
});

export default entry;

function register(api: OpenClawPluginApi): void {
  const config = loadConfig(api.pluginConfig ?? {});
  if (!config.enabled) {
    api.logger.info("[mindmemos] plugin disabled by config");
    return;
  }

  api.on("before_prompt_build", async (event: unknown, ctx: unknown) => {
    const query = extractQuery(event);
    const sessionId = resolveSessionId(config, event, ctx);
    if (!query || query.trim().length < config.minQueryLength) {
      return {};
    }

    try {
      const result = await searchMemories(config, query, sessionId);
      const context = formatMemoryContext(result.memories ?? [], config.userId, isSubagentSession(sessionId));
      if (!context) {
        return {};
      }
      api.logger.info(`[mindmemos] recall hit ${result.memories?.length ?? 0} memories, injected ${context.length} chars`);
      return { prependContext: context };
    } catch (error) {
      api.logger.warn(`[mindmemos] memory search failed: ${errorMessage(error)}`);
      return {};
    }
  });

  api.on("agent_end", async (event, ctx) => {
    if (!isSuccessfulRun(event)) {
      return;
    }
    const turn = pickLastTurnMessages(event);
    if (turn.length === 0) {
      return;
    }
    const sessionId = resolveSessionId(config, event, ctx, turn);
    const turnId = nonemptyText(getPath(event, ["runId"])) ?? nonemptyText(getPath(ctx, ["runId"]));
    const agentId = nonemptyText(ctx.agentId) ?? config.appId;
    const eventId = `openclaw:${turnId ?? randomUUID()}`;

    try {
      const skillVersions = await loadedSkillVersions(config.cli, turn, ctx.workspaceDir, {
        warn: (message) => api.logger.warn(`[mindmemos] ${message}`),
      });
      await reportTrajectory(config.cli, {
        event_id: eventId,
        source: "openclaw-plugin",
        messages: turn.slice(-config.maxConversationMessages),
        skill_version_ids: skillVersions,
        user_id: config.userId,
        agent_id: agentId,
        app_id: config.appId,
        session_id: sessionId,
        turn_id: turnId,
        memory_mode: config.addMode,
      }, {
        info: (message) => api.logger.info(`[mindmemos] ${message}`),
        warn: (message) => api.logger.warn(`[mindmemos] ${message}`),
      });
    } catch (error) {
      api.logger.warn(`[mindmemos] trajectory ${eventId}: ${errorMessage(error)}`);
    }
  });

  api.logger.info("[mindmemos] plugin loaded");
}

function loadConfig(raw: unknown): PluginConfig {
  const obj = isRecord(raw) ? raw : {};
  const addMode = obj.addMode === "async" ? "async" : DEFAULT_CONFIG.addMode;
  return {
    enabled: obj.enabled !== false,
    cli: typeof obj.cli === "string" && obj.cli.trim() ? obj.cli.trim() : DEFAULT_CONFIG.cli,
    topK: positiveInteger(obj.topK, DEFAULT_CONFIG.topK),
    multimodal: obj.multimodal !== false,
    tokenBudget: optionalPositiveInteger(obj.tokenBudget),
    addMode,
    appId: typeof obj.appId === "string" && obj.appId.trim() ? obj.appId.trim() : DEFAULT_CONFIG.appId,
    sessionId: typeof obj.sessionId === "string" && obj.sessionId.trim() ? obj.sessionId.trim() : undefined,
    userId: typeof obj.userId === "string" && obj.userId.trim() ? obj.userId.trim() : undefined,
    minQueryLength: positiveInteger(obj.minQueryLength, DEFAULT_CONFIG.minQueryLength),
    maxConversationMessages: positiveInteger(
      obj.maxConversationMessages,
      DEFAULT_CONFIG.maxConversationMessages,
    ),
  };
}

async function searchMemories(config: PluginConfig, query: string, sessionId: string): Promise<MemorySearchResult> {
  const args = ["memory", "search", query, "--top-k", String(config.topK), "--json"];
  if (config.multimodal) {
    // Image hits carry metadata.blob_id + a short-lived image_url the agent can GET directly;
    // the server silently ignores this flag while the project's multimodal mode is off.
    args.push("--multimodal");
  }
  if (config.tokenBudget) {
    args.push("--token-budget", String(config.tokenBudget));
  }
  args.push("--app-id", config.appId, "--session-id", sessionId);
  if (config.userId) {
    args.push("--user-id", config.userId);
  }
  return spawnFileJson<MemorySearchResult>(config.cli, args);
}

function formatMemoryContext(memories: MemoryHit[], userId: string | undefined, isSubagent: boolean): string {
  let hasImage = false;
  const lines = memories
    .map((hit, index) => {
      const text = typeof hit.memory === "string" ? hit.memory.trim() : "";
      const imageUrl =
        typeof hit.metadata?.image_url === "string" && hit.metadata.image_url
          ? hit.metadata.image_url
          : undefined;
      if (!text && !imageUrl) {
        return null;
      }
      hasImage = hasImage || Boolean(imageUrl);
      const label = hit.id ? `${index + 1}. [${hit.id}]` : `${index + 1}.`;
      const when = hit.last_update_at ?? hit.event_time ?? hit.source_timestamp;
      const body = [text || "[image memory]", imageUrl ? `<image: ${imageUrl}>` : ""].filter(Boolean).join(" ");
      return when ? `${label} ${body} (${when})` : `${label} ${body}`;
    })
    .filter((line): line is string => line !== null);

  if (lines.length === 0) {
    return "";
  }
  const preamble = buildPreamble(userId, isSubagent);
  const imageHint = hasImage
    ? "(<image: …> URLs point at the stored images, fetch directly with no API key, and expire; relative paths resolve against the MindMemOS API base)"
    : "";
  return [MEMORY_CONTEXT_OPEN, preamble, imageHint, ...lines, MEMORY_CONTEXT_CLOSE].filter(Boolean).join("\n");
}

/**
 * One-line instruction placed before the recalled memories. For a subagent the
 * memories are someone else's; we frame them as background only and tell the
 * model not to take on that user's identity.
 */
function buildPreamble(userId: string | undefined, isSubagent: boolean): string {
  if (isSubagent) {
    const who = userId ? `user "${userId}"` : "the user";
    return `The following stored memories for ${who} are provided as background context only. You are a subagent — use them for context but do not assume you are this user.`;
  }
  return userId ? `Relevant memories for ${userId}:` : "Relevant memories:";
}

/** Subagent sessions carry a `:subagent:` segment in their session key. */
function isSubagentSession(sessionId: string): boolean {
  return /:subagent:/i.test(sessionId);
}

/**
 * The recall query for the current turn. In `before_prompt_build` the live user
 * input lives on `event.prompt`; `event.messages` only holds prior session
 * history (empty on the first turn, stale afterwards). Prefer `prompt` and fall
 * back to scanning messages only when it is absent.
 */
function extractQuery(event: unknown): string {
  const prompt = getPath(event, ["prompt"]);
  if (typeof prompt === "string" && prompt.trim()) {
    return prompt.trim();
  }
  return extractLatestUserText(event);
}

function extractLatestUserText(event: unknown): string {
  const messages = getMessagesArray(event);
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = normalizeMessage(messages[i]);
    if (message?.role === "user" && !isInternalMemoryMessage(message)) {
      return message.content;
    }
  }

  const directMessages = [
    getPath(event, ["message"]),
    getPath(event, ["latestMessage"]),
    getPath(event, ["turn", "message"]),
    getPath(event, ["request", "message"]),
  ];
  for (const candidate of directMessages) {
    const message = normalizeMessage(candidate);
    if (message?.role === "user" && !isInternalMemoryMessage(message)) {
      return message.content;
    }
  }

  return "";
}

/** Whether the finished run succeeded; missing flag is treated as success. */
function isSuccessfulRun(event: unknown): boolean {
  return getPath(event, ["success"]) !== false;
}

/** Select the last real user message and the assistant/tool messages that follow it. */
function pickLastTurnMessages(event: unknown): TrajectoryMessage[] {
  const all = getMessagesArray(event)
    .map(normalizeMessage)
    .filter((message): message is TrajectoryMessage => message !== null && !isInternalMemoryMessage(message));
  const lastUserIndex = all.map((message) => message.role).lastIndexOf("user");
  return lastUserIndex < 0 ? [] : all.slice(lastUserIndex);
}

function resolveSessionId(
  config: PluginConfig,
  event: unknown,
  ctx: unknown,
  messages: TrajectoryMessage[] = [],
): string {
  if (config.sessionId) {
    return sanitizeSessionId(config.sessionId);
  }

  // The agent_end event payload does not carry a session id; the canonical
  // session key lives on the hook context (`ctx.sessionKey`). Prefer it so all
  // turns of one conversation share a session_id on the mindmemos side.
  const candidates = [
    getPath(ctx, ["sessionKey"]),
    getPath(ctx, ["sessionId"]),
    getPath(ctx, ["session_id"]),
    getPath(ctx, ["session", "id"]),
    getPath(event, ["sessionId"]),
    getPath(event, ["session_id"]),
    getPath(event, ["session", "id"]),
    getPath(event, ["session", "sessionId"]),
    getPath(event, ["session", "session_id"]),
    getPath(event, ["conversationId"]),
    getPath(event, ["conversation_id"]),
    getPath(event, ["conversation", "id"]),
    getPath(event, ["conversation", "conversationId"]),
    getPath(event, ["threadId"]),
    getPath(event, ["thread_id"]),
    getPath(event, ["thread", "id"]),
    getPath(event, ["request", "sessionId"]),
    getPath(event, ["request", "session_id"]),
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return sanitizeSessionId(candidate.trim());
    }
    if (typeof candidate === "number" && Number.isFinite(candidate)) {
      return String(candidate);
    }
  }

  // Fallback: derive a stable, NUL-free id from the first message. The raw
  // signature contains NUL separators, so hash it — passing NUL bytes as a
  // CLI argument makes child_process.spawn throw and the add silently fail.
  if (messages.length > 0) {
    const digest = createHash("sha1").update(messageSignature(messages[0])).digest("hex");
    return `openclaw:${digest}`;
  }
  return "openclaw:default";
}

/** Strip control characters (notably NUL) so the value is safe as a CLI argument. */
function sanitizeSessionId(value: string): string {
  // eslint-disable-next-line no-control-regex
  const cleaned = value.replace(/[\u0000-\u001f\u007f]/g, "");
  return cleaned || "openclaw:default";
}

function getMessagesArray(event: unknown): unknown[] {
  const candidates = [
    getPath(event, ["messages"]),
    getPath(event, ["conversation", "messages"]),
    getPath(event, ["session", "messages"]),
    getPath(event, ["finalMessages"]),
    getPath(event, ["transcript"]),
  ];
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) {
      return candidate;
    }
  }
  return [];
}

function normalizeMessage(raw: unknown): TrajectoryMessage | null {
  if (!isRecord(raw)) {
    return null;
  }

  const role = normalizeRole(raw.role ?? raw.type ?? raw.sender);
  const rawContent = raw.content ?? raw.text ?? raw.message;
  const content = textFromUnknown(rawContent);
  const toolCalls = role === "assistant" && Array.isArray(rawContent)
    ? rawContent.map(parseToolCall).filter((call): call is ToolCall => call !== undefined)
    : [];
  const toolCallId = role === "tool" ? nonemptyText(raw.toolCallId) : undefined;
  if (!role || (!content && toolCalls.length === 0 && !toolCallId)) {
    return null;
  }

  const details = asRecord(raw.details);
  const toolError = raw.isError === true || details?.status === "failed" || details?.status === "running"
    || (typeof details?.exitCode === "number" && details.exitCode !== 0)
    || /\(Command exited with code [1-9]\d*\)\s*$/.test(content);
  return {
    role,
    content,
    timestamp: timestampMillis(raw.timestamp ?? raw.createdAt ?? raw.time),
    ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
    ...(toolCallId ? { tool_call_id: toolCallId } : {}),
    ...(role === "tool" ? { is_error: toolError, [skillEvidenceText]: textFromUnknown(rawContent, true) } : {}),
  };
}

function nonemptyText(raw: unknown): string | undefined {
  return typeof raw === "string" ? raw.trim() || undefined : undefined;
}

function messageSignature(message: TrajectoryMessage): string {
  return `${message.timestamp}\u0000${message.role}\u0000${message.content}`;
}

function isInternalMemoryMessage(message: TrajectoryMessage): boolean {
  const text = message.content.trim();
  if (!text) {
    return !message.tool_call_id && !message.tool_calls?.length;
  }
  if (text.startsWith(MEMORY_CONTEXT_OPEN) && text.endsWith(MEMORY_CONTEXT_CLOSE)) {
    return true;
  }
  if (text === "HEARTBEAT_OK" || text === "[OpenClaw heartbeat poll]") {
    return true;
  }
  return (
    text.includes("An async command completion event was triggered") ||
    text.includes("user delivery is disabled for this run") ||
    text.includes("reply HEARTBEAT_OK only") ||
    text.includes("Do not mention, summarize, or reuse command output")
  );
}

function normalizeRole(raw: unknown): TrajectoryMessage["role"] | null {
  if (typeof raw !== "string") {
    return null;
  }
  const role = raw.toLowerCase();
  if (role === "human") {
    return "user";
  }
  if (role === "ai" || role === "model") {
    return "assistant";
  }
  // OpenClaw tool result messages carry the role "toolResult"; map them to the
  // generic "tool" role so they are persisted as tool replies.
  if (role === "toolresult" || role === "tool_result") {
    return "tool";
  }
  if (role === "user" || role === "assistant" || role === "system" || role === "tool") {
    return role;
  }
  return null;
}

function textFromUnknown(value: unknown, preserveWhitespace = false): string {
  if (typeof value === "string") {
    return preserveWhitespace ? value : value.trim();
  }
  if (Array.isArray(value)) {
    const text = value.map((item) => textFromUnknown(item, preserveWhitespace)).filter(Boolean).join("\n");
    return preserveWhitespace ? text : text.trim();
  }
  if (isRecord(value)) {
    const toolCall = textFromToolCall(value);
    if (toolCall) {
      return toolCall;
    }
    return textFromUnknown(value.text ?? value.content ?? value.value, preserveWhitespace);
  }
  return "";
}

/**
 * Render an OpenClaw assistant tool-call content block as text. The block looks
 * like `{ type: "toolCall", name, arguments }` and carries no text/content
 * field, so without this it would serialize to an empty string and the call
 * would be lost. The tool call stays on the assistant message; we emit a single
 * line `[tool_call] name(<json args>)`.
 */
function textFromToolCall(value: Record<string, unknown>): string {
  const call = parseToolCall(value);
  if (!call) {
    return "";
  }
  const args = typeof call.arguments === "string" ? call.arguments : JSON.stringify(call.arguments);
  return `[tool_call] ${call.name}(${args})`;
}

function parseToolCall(raw: unknown): ToolCall | undefined {
  const value = asRecord(raw);
  const name = nonemptyText(value?.name);
  if (!value || !name || value.type !== "toolCall") {
    return undefined;
  }
  const id = nonemptyText(value.id);
  const args = value.arguments;
  return {
    ...(id ? { id } : {}),
    name,
    arguments: typeof args === "string" ? args : asRecord(args) ?? {},
  };
}

function timestampMillis(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value > 10_000_000_000 ? Math.round(value) : Math.round(value * 1000);
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

function getPath(value: unknown, path: string[]): unknown {
  let cursor = value;
  for (const key of path) {
    if (!isRecord(cursor)) {
      return undefined;
    }
    cursor = cursor[key];
  }
  return cursor;
}

function positiveInteger(value: unknown, fallback: number): number {
  return optionalPositiveInteger(value) ?? fallback;
}

function optionalPositiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
