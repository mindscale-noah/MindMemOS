import { spawn } from "node:child_process";
import { basename, resolve } from "node:path";

export const skillEvidenceText = Symbol("skillEvidenceText");

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
  [skillEvidenceText]?: string;
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

type SkillReadEvidence = { kind: "read"; path: string; content: string };
type SkillEvidence = SkillReadEvidence | { kind: "shell"; command: string; content: string };

/** Send only observed loads to the SDK, which owns Skill hashing and resolution. */
export async function loadedSkillVersions(
  cli: string,
  messages: readonly TrajectoryMessage[],
  cwd: string | undefined,
  logger: { warn: (message: string) => void },
  nativeReads: readonly SkillReadEvidence[] = [],
): Promise<string[]> {
  const evidence: SkillEvidence[] = [...toolSkillEvidence(messages, cwd), ...nativeReads];
  if (!evidence.length) {
    return [];
  }
  try {
    const result = await spawnFile(cli, ["skill", "identify", "--file", "-"], JSON.stringify(evidence), 10_000);
    const versions = parseObject(result.stdout)?.skill_version_ids;
    if (result.code === 0 && Array.isArray(versions) && versions.every(isVersionId)) {
      return versions;
    }
  } catch {
    // An unavailable resolver must not suppress the Memory report.
  }
  logger.warn("Skill identification did not return version IDs; continuing with Memory reporting");
  return [];
}

function toolSkillEvidence(messages: readonly TrajectoryMessage[], cwd?: string): SkillEvidence[] {
  const pending = new Map<string, ToolCall>();
  const evidence: SkillEvidence[] = [];
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
    const content = message[skillEvidenceText] ?? message.content;
    const command = args?.command;
    if (["exec", "bash", "shell"].includes(tool) && typeof command === "string" && /mindmemos/i.test(command)) {
      evidence.push({ kind: "shell", command, content });
      continue;
    }
    const path = args?.path ?? args?.file_path ?? args?.filepath;
    if (["read", "read_file"].includes(tool) && typeof path === "string" && basename(path) === "SKILL.md" && content.trim()) {
      evidence.push({ kind: "read", path: cwd ? resolve(cwd, path) : path, content });
    }
  }
  return evidence;
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

function spawnFile(command: string, args: string[], stdin?: string, timeout?: number): Promise<SpawnResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["pipe", "pipe", "pipe"],
      timeout,
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
