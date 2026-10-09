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
  const response = await fetch("/api/" + path, {
    method,
    headers: { "Content-Type": "application/json", "X-Workbench-Token": token },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const messages: Record<number, string> = {
      401: "Authentication failed. Check your API key.",
      403: "Access denied or session expired. Refresh the page.",
      409:
        path.startsWith("service-config") ||
        path.startsWith("configurations") ||
        path.startsWith("client-configurations")
          ? "The configuration file has changed. Copy your draft, then reload before saving."
          : "Conflict: the version label is already in use, the contents have changed, or the Skill is still referenced.",
      429: "操作频繁，请稍后重试。",
      503: path.startsWith("memories")
        ? "The Memory service is not configured or is temporarily unavailable. Check the connection settings."
        : "The Skill service is not configured or is temporarily unavailable. Check the connection settings.",
    };
    throw new Error(
      (response.status === 429 ? messages[429] : data.detail) ||
        messages[response.status] ||
        `Request failed (${response.status})`,
    );
  }
  return response.json();
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
