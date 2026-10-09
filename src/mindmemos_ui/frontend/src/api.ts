let token = "";
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
      401: "认证失败，请检查 API Key。",
      403: "没有权限或会话已失效，请刷新页面。",
      409:
        path.startsWith("service-config") ||
        path.startsWith("configurations") ||
        path.startsWith("client-configurations")
          ? "配置文件已被修改。请先复制草稿，再重新读取后保存。"
          : "操作冲突：版本号已被占用、内容已变化，或 Skill 仍被引用。",
      429: "请求过于频繁，请稍后重试。",
      503: path.startsWith("memories")
        ? "Memory 服务未配置或暂时不可用，请检查连接设置。"
        : "Skill 服务未配置或暂时不可用，请检查连接设置。",
    };
    throw new Error(
      data.detail ||
        messages[response.status] ||
        `请求失败 (${response.status})`,
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
  fallback = "未命名版本",
) {
  const label = version?.version_label || fallback;
  const implementation = version?.implementation;
  return implementation && implementation !== "static"
    ? `${label}-dynamic-${implementation}`
    : label;
}
