import { requestErrorMessage } from "./errors";
let token = "";
export type TrajectoryCounts = {
  total: number;
  consumed: number;
  unconsumed: number;
};
export type TrajectoryStatistics = {
  skill_version_id: string;
  timezone: string;
  totals: TrajectoryCounts;
  days: (TrajectoryCounts & { day: string })[];
  last_evolved_at: string | null;
};
export type Version = {
  implementation?: string;
  version_id: string;
  version_label?: string | null;
  commit_message?: string | null;
  status: string;
  origin: string;
  created_at: string;
};
export type Skill = {
  implementation?: string;
  skill_id: string;
  name: string;
  active_version_id: string;
  active_version_label?: string | null;
  latest_version_id?: string;
  latest_version_label?: string | null;
  latest_implementation?: string;
  created_at?: string;
  alias?: string;
};
export type Ref = {
  source: "local" | "graph";
  skill_id: string;
  version_id: string;
};
export type Selection = {
  source: "local" | "graph";
  id: string;
  name: string;
  local?: Skill;
  version?: string;
};
export type ConnectionStatus =
  | "connected"
  | "unconfigured"
  | "unauthorized"
  | "unavailable";
export type Connections = Record<
  "memory" | "skill",
  { url: string; status: ConnectionStatus }
>;
export async function api<T>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch("/api/" + path, {
      method,
      headers: {
        "Content-Type": "application/json",
        "X-Workbench-Token": token,
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new Error(
      "Cannot connect to the UI server. Make sure it is running, then refresh the page.",
    );
  }
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(requestErrorMessage(path, response.status, data?.detail));
  }
  try {
    return await response.json();
  } catch {
    throw new Error("The server returned an invalid response. Please retry.");
  }
}
export async function session() {
  const result = await api<{
    token: string;
    home: string;
    graphs_connected: boolean;
  }>("session");
  token = result.token;
  return result;
}
export function query(values: Record<string, string>) {
  return new URLSearchParams(values).toString();
}
export function ref(selection: Selection, version: string): Ref {
  return {
    source: selection.source,
    skill_id: selection.id,
    version_id: version,
  };
}

export const versionPattern =
  "v(0|[1-9][0-9]*)[.](0|[1-9][0-9]*)[.](0|[1-9][0-9]*)";

export function versionLabel(
  version?: { version_label?: string | null; implementation?: string },
  fallback = "Unnamed version",
) {
  const label = version?.version_label || fallback;
  const implementation = version?.implementation;
  return implementation && implementation !== "static"
    ? `${label}-dynamic-${implementation}`
    : label;
}
