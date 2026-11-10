import { spawn } from "node:child_process";
import { dirname, isAbsolute, join, resolve, sep } from "node:path";

export type ToolCall = {
  id?: string;
  name: string;
  arguments: Record<string, unknown> | string;
};

export type TrajectoryMessage = {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  timestamp?: number;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  is_error?: boolean;
};

export type TrajectoryEvent = {
  event_id: string;
  source: string;
  messages: TrajectoryMessage[];
  skill_version_ids: string[];
  user_id?: string;
  agent_id: string;
  app_id: string;
  session_id: string;
  turn_id?: string;
  started_at?: string;
  finished_at?: string;
  memory_mode: "sync" | "async";
};

type Delivery = {
  status: "queued" | "completed" | "stored" | "skipped" | "failed" | "unknown";
  code?: string;
  request_id?: string;
  trajectory_id?: string;
  http_status?: number;
};

type SpawnResult = {
  stdout: string;
  stderr: string;
  code: number | null;
};

export async function spawnFileJson<T>(command: string, args: string[]): Promise<T> {
  const result = await spawnFile(command, args);
  if (result.code !== 0) {
    throw new Error(`mindmemos exited with code ${result.code}: ${result.stderr || result.stdout}`.trim());
  }
  return JSON.parse(result.stdout) as T;
}

/** CLI exit 1 can still carry one successful delivery and one failure. */
export async function reportTrajectory(
  command: string,
  event: TrajectoryEvent,
  logger: { info: (message: string) => void; warn: (message: string) => void },
): Promise<void> {
  const result = await spawnFile(command, ["trajectory", "report", "--file", "-", "--json"], JSON.stringify(event));
  const receipt = parseObject(result.stdout);
  if (receipt?.event_id !== event.event_id || !isDelivery(receipt.memory) || !isDelivery(receipt.skill)) {
    const error = parseObject(result.stderr);
    const code = typeof error?.code === "string" ? `, ${error.code}` : "";
    throw new Error(
      `no delivery receipt (CLI exit ${result.code}${code}); check \`mindmemos trajectory report --help\` and SDK setup`,
    );
  }
  for (const [destination, delivery] of [["Memory", receipt.memory], ["Skill", receipt.skill]] as const) {
    const details = [delivery.code, delivery.http_status, delivery.request_id, delivery.trajectory_id]
      .filter(Boolean)
      .join(" ");
    const message = `trajectory ${event.event_id}: ${destination} ${delivery.status}${details ? ` (${details})` : ""}`;
    if (delivery.status === "failed" || delivery.status === "unknown") {
      logger.warn(message);
    } else {
      logger.info(message);
    }
  }
}

/** Attribute only completed reads or shows paired with their own tool results. */
export function loadedSkillVersions(messages: readonly TrajectoryMessage[], cwd?: string): string[] {
  const pending = new Map<string, ToolCall>();
  const exported = new Map<string, string>();
  const loaded = new Set<string>();
  for (const message of messages) {
    for (const call of message.tool_calls ?? []) {
      if (call.id) {
        pending.set(call.id, call);
      }
    }
    if (message.role !== "tool" || !message.tool_call_id) {
      continue;
    }
    const call = pending.get(message.tool_call_id);
    pending.delete(message.tool_call_id);
    if (!call || message.is_error) {
      continue;
    }
    const args = typeof call.arguments === "string" ? parseObject(call.arguments) : call.arguments;
    const tool = call.name.toLowerCase();
    const command = args?.command;
    if (["exec", "bash"].includes(tool) && typeof command === "string") {
      // Require a CLI invocation, not arbitrary output that happens to contain a version ID.
      const action = command.match(/^\s*(?:uv\s+run\s+)?mindmemos\s+skill\s+(export|show)\s+[^;&|\r\n`<>]+$/)?.[1];
      if (action === "export") {
        const result = parseObject(message.content);
        if (typeof result?.directory === "string" && isAbsolute(result.directory) && isVersionId(result.skill_version_id)) {
          exported.set(join(result.directory, "SKILL.md"), result.skill_version_id);
        }
      } else if (action === "show") {
        const version = parseObject(message.content)?.version;
        if (isRecord(version) && isVersionId(version.skill_version_id) && typeof version.content === "string" && version.content.trim()) {
          loaded.add(version.skill_version_id);
        }
      }
      continue;
    }
    const path = args?.path ?? args?.file_path ?? args?.filepath;
    if (typeof path !== "string" || (!cwd && !isAbsolute(path))) {
      continue;
    }
    const file = resolve(cwd ?? "", path);
    const version = exported.get(file);
    if (tool === "read" && version && message.content.trim()) {
      loaded.add(version);
    } else if (tool === "write" || tool === "edit") {
      // A subsequent read may see modified content. Invalidate the whole exported bundle.
      for (const document of exported.keys()) {
        const directory = dirname(document);
        if (file === directory || file.startsWith(directory + sep)) {
          exported.delete(document);
        }
      }
    }
  }
  return [...loaded];
}

export function asRecord(value: unknown): Record<string, unknown> | undefined {
  return isRecord(value) ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isVersionId(value: unknown): value is string {
  return typeof value === "string" && /^(?:skv-|v_)[A-Za-z0-9_-]+$/.test(value);
}

function isDelivery(value: unknown): value is Delivery {
  return isRecord(value) && typeof value.status === "string"
    && ["queued", "completed", "stored", "skipped", "failed", "unknown"].includes(value.status);
}

function parseObject(json: string): Record<string, unknown> | undefined {
  try {
    return asRecord(JSON.parse(json));
  } catch {
    return undefined;
  }
}

function spawnFile(command: string, args: string[], stdin?: string): Promise<SpawnResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf-8").on("data", (chunk: string) => { stdout += chunk; });
    child.stderr.setEncoding("utf-8").on("data", (chunk: string) => { stderr += chunk; });
    child.on("error", reject);
    // The CLI can exit before reading stdin, for example if it predates this command.
    child.stdin.on("error", (error: NodeJS.ErrnoException) => {
      if (error.code !== "EPIPE") {
        reject(error);
      }
    });
    child.on("close", (code) => resolve({ stdout, stderr, code }));
    child.stdin.end(stdin);
  });
}
