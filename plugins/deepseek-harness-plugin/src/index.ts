import z from "@deepseek-ai/schemastery";
import { createUserMessage } from "@deepseek-ai/dsh-llm";
import type { Context } from "@deepseek-ai/cordis";
import type { Agent } from "@deepseek-ai/dsh-agent";
import type { ContentBlock, UserMessage } from "@deepseek-ai/dsh-llm";
import type { Session, SessionEvent } from "@deepseek-ai/dsh-session";
import {
  loadedSkillVersions, reportTrajectory, spawnFileJson,
  type ToolCall, type TrajectoryMessage,
} from "./generated/mindmemos-cli.js";
import { registerBundledSkill } from "./skill-provider.js";

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

/** Cordis plugin name, also stamped on injected messages for provenance. */
export const name = "mindmemos-memory";

/** Host services for recall and registration of the bundled Skill. */
export const inject = ["agents", "skills"];

/** Resolved plugin configuration, filled in by the {@link Config} schema before {@link apply} runs. */
export interface Config {
  /** Executable used to invoke the CLI — a name on `PATH` or an absolute path, not a shell command with arguments. */
  cli: string;
  /** Number of memories injected per turn. */
  topK: number;
  /** Search stored images too; image hits carry a short-lived directly-fetchable image_url. */
  multimodal: boolean;
  /** Optional strict token budget for recall; enables token-budget retention when set. */
  tokenBudget?: number;
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
}

/** Schemastery schema for {@link Config}; cordis validates and defaults the entry config with it. */
export const Config: z<Config> = z.object({
  cli: z.string().default("mindmemos"),
  topK: z.natural().min(1).default(5),
  multimodal: z.boolean().default(true),
  tokenBudget: z.natural().min(1),
  // sync by default: async needs kafka.enabled=true on the server — without Kafka the
  // add is rejected with a clean 400 (add.async_unavailable) rather than stored;
  // sync works on every deployment.
  addMode: z.union([z.const("sync"), z.const("async")]).default("sync"),
  userId: z.string(),
  appId: z.string().default("deepseek-harness"),
  sessionId: z.string(),
  minQueryLength: z.natural().min(1).default(2),
  maxConversationMessages: z.natural().min(1).default(80),
});

export function apply(ctx: Context, config: Config): void {
  registerBundledSkill(ctx);

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

  ctx.on("session/event", (session, event) => {
    if (event.type !== "turn/end" || event.data.reason.kind !== "completed") {
      return;
    }
    const turn = collectTurnMessages(session, event);
    if (!turn || turn.messages.length === 0) {
      return;
    }

    const hostSessionId = String(session.id);
    const turnId = String(event.data.turn);
    const eventId = `deepseek-harness:${hostSessionId}:${turnId}`;
    const logger = ctx.logger(name);
    void reportTrajectory(config.cli, {
      event_id: eventId,
      source: "deepseek-harness-plugin",
      messages: turn.messages.slice(-config.maxConversationMessages),
      skill_version_ids: loadedSkillVersions(turn.messages, session.header.cwd),
      user_id: config.userId?.trim() || undefined,
      agent_id: session.header.agentPreset ?? config.appId,
      app_id: config.appId,
      session_id: resolveSessionId(config, hostSessionId),
      turn_id: turnId,
      started_at: new Date(turn.startTime).toISOString(),
      finished_at: new Date(event.time).toISOString(),
      memory_mode: config.addMode,
    }, {
      info: (message) => logger.info(message),
      warn: (message) => logger.warn(message),
    }).catch((error: unknown) => {
      logger.warn(`trajectory ${eventId}: ${errorMessage(error)}`);
    });
  });

  ctx.logger(name).info("plugin loaded");
}

async function searchMemories(config: Config, query: string, sessionId: string): Promise<MemorySearchResult> {
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

function formatMemoryContext(memories: MemoryHit[], userId: string | undefined): string {
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
  const preamble = userId ? `Relevant memories for ${userId}:` : "Relevant memories:";
  const imageHint = hasImage
    ? "(<image: …> URLs point at the stored images, fetch directly with no API key, and expire; relative paths resolve against the MindMemOS API base)"
    : "";
  return [MEMORY_CONTEXT_OPEN, preamble, imageHint, ...lines, MEMORY_CONTEXT_CLOSE].filter(Boolean).join("\n");
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

/** Read only the events enclosed by this turn's start and end. */
function collectTurnMessages(
  session: Session,
  end: SessionEvent<"turn/end">,
): { messages: TrajectoryMessage[]; startTime: number } | undefined {
  const messages: TrajectoryMessage[] = [];
  for (let i = session.events.length - 1; i >= 0; i -= 1) {
    const event = session.events[i];
    if (event.seq >= end.seq) {
      continue;
    }
    if (event.type === "turn/start") {
      return event.data.turn === end.data.turn
        ? { messages: messages.reverse(), startTime: event.time }
        : undefined;
    }
    const message = messageFromEvent(event);
    if (message) {
      messages.push(message);
    }
  }
  return undefined;
}

function messageFromEvent(event: SessionEvent): TrajectoryMessage | null {
  switch (event.type) {
    case "user/message": {
      // Plugin and Skill snapshots are context, not fresh user dialogue.
      if (event.data.source.kind !== "user") {
        return null;
      }
      const content = textFromContent(event.data.content);
      return content ? { role: "user", content, timestamp: event.time } : null;
    }
    case "assistant/message": {
      const blocks = event.data.message.content;
      const content = textFromContent(blocks);
      const toolCalls = toolCallsFromContent(blocks);
      return content || toolCalls.length
        ? { role: "assistant", content, timestamp: event.time, ...(toolCalls.length ? { tool_calls: toolCalls } : {}) }
        : null;
    }
    case "tool/result": {
      const message = event.data.message;
      return {
        role: "tool",
        content: textFromContent(message.content),
        timestamp: event.time,
        tool_call_id: String(message.source.callId),
        is_error: message.content[0]?.isError === true || event.data.error !== undefined,
      };
    }
    default:
      return null;
  }
}

function toolCallsFromContent(content: readonly ContentBlock[]): ToolCall[] {
  return content.flatMap((block) => block.type === "tool-call" ? [{
    id: String(block.id),
    name: block.name,
    arguments: block.arguments,
  }] : []);
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
