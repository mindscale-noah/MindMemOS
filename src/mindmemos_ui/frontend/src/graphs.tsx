import { useState, useEffect } from "react";
import { api, type Selection } from "./api";
import { useLoad, Message, Empty } from "./components";
type GraphNode = {
  node_id: string;
  skill_id: string;
  skill_version_id: string;
  name: string;
  version_label?: string;
};
type Edge = {
  edge_id: string;
  source_node_id: string;
  target_node_id: string;
  edge_type: string;
};
export function Graphs({
  connected,
  open,
}: {
  connected: boolean;
  open: (s: Selection) => void;
}) {
  const [graph, setGraph] = useState("");
  const [snapshot, setSnapshot] = useState("");
  const [node, setNode] = useState("");
  const [search, setSearch] = useState("");
  const [edgeType, setEdgeType] = useState("");
  const [reload, setReload] = useState(0);
  const graphs = useLoad<any[]>(
    () => (connected ? api("graphs") : Promise.resolve([])),
    "graphs" + reload,
  );
  const snapshots = useLoad<any[]>(
    () =>
      graph
        ? api(`graphs/${encodeURIComponent(graph)}/snapshots`)
        : Promise.resolve([]),
    graph + reload,
  );
  const data = useLoad<{ nodes: GraphNode[]; edges: Edge[] }>(
    () =>
      snapshot
        ? api(`graphs/${encodeURIComponent(graph)}/snapshots/${snapshot}`)
        : Promise.resolve({ nodes: [], edges: [] }),
    graph + snapshot + reload,
  );
  useEffect(() => {
    if (graphs.value?.length && !graph) setGraph(graphs.value[0].graph_id);
  }, [graphs.value]);
  useEffect(() => {
    const rows = snapshots.value;
    if (rows?.length) {
      const latest = [...rows].sort(
        (a, b) =>
          Math.max(0, ...b.revisions) - Math.max(0, ...a.revisions) ||
          b.created_at.localeCompare(a.created_at),
      )[0];
      setSnapshot(latest.graph_version_id);
    }
  }, [snapshots.value]);
  const nodes = data.value?.nodes || [];
  const edges = (data.value?.edges || []).filter(
    (e) => !edgeType || e.edge_type === edgeType,
  );
  const selected = nodes.find((n) => n.node_id === node);
  const points = new Map(
    nodes.map((n, i) => [
      n.node_id,
      { x: 145 + (i % 3) * 270, y: 65 + Math.floor(i / 3) * 130 },
    ]),
  );
  const names = new Map(
    nodes.map((n) => [
      n.node_id,
      `${n.name}@${n.version_label || "未命名版本"}`,
    ]),
  );
  return (
    <>
      <div className="row">
        <h1>管理图谱</h1>
        <button onClick={() => setReload(reload + 1)}>刷新</button>
      </div>
      {!connected ? (
        <Empty>无法连接工作区数据库，请检查本地工作区配置。</Empty>
      ) : (
        <>
          <div className="fields">
            <label>
              图谱
              <select
                value={graph}
                onChange={(e) => {
                  setGraph(e.target.value);
                  setSnapshot("");
                  setNode("");
                }}
              >
                <option value="">选择图谱</option>
                {graphs.value?.map((g) => (
                  <option key={g.graph_id} value={g.graph_id}>
                    {g.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              图谱快照
              <select
                value={snapshot}
                onChange={(e) => {
                  setSnapshot(e.target.value);
                  setNode("");
                }}
              >
                <option value="">选择快照</option>
                {snapshots.value?.map((s) => (
                  <option key={s.graph_version_id} value={s.graph_version_id}>
                    {s.revisions.length
                      ? "修订 " + s.revisions.join(", ")
                      : s.status}{" "}
                    · {s.graph_version_id.slice(0, 8)}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <Message text={graphs.error || snapshots.error || data.error} error />
          {nodes.length ? (
            <>
              <div className="toolbar">
                <input
                  placeholder="定位 Skill"
                  aria-label="定位 Skill"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                <select
                  aria-label="关系类型"
                  value={edgeType}
                  onChange={(e) => setEdgeType(e.target.value)}
                >
                  <option value="">全部关系</option>
                  {[...new Set(data.value?.edges.map((e) => e.edge_type))].map(
                    (t) => (
                      <option key={t}>{t}</option>
                    ),
                  )}
                </select>
              </div>
              <div className="graph-canvas">
                <svg
                  width="830"
                  height={Math.ceil(nodes.length / 3) * 130 + 20}
                  aria-label="Skill 组织关系图"
                >
                  <defs>
                    <marker
                      id="arrow"
                      viewBox="0 0 10 10"
                      refX="10"
                      refY="5"
                      markerWidth="6"
                      markerHeight="6"
                      orient="auto-start-reverse"
                    >
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#9fa6b2" />
                    </marker>
                  </defs>
                  {edges.map((e) => {
                    const a = points.get(e.source_node_id),
                      b = points.get(e.target_node_id);
                    if (!a || !b) return null;
                    return (
                      <line
                        key={e.edge_id}
                        x1={
                          a.y === b.y ? a.x + Math.sign(b.x - a.x) * 120 : a.x
                        }
                        y1={a.y === b.y ? a.y : a.y + Math.sign(b.y - a.y) * 35}
                        x2={
                          a.y === b.y ? b.x - Math.sign(b.x - a.x) * 120 : b.x
                        }
                        y2={a.y === b.y ? b.y : b.y - Math.sign(b.y - a.y) * 35}
                        stroke="#bbc2cc"
                        markerEnd="url(#arrow)"
                      />
                    );
                  })}
                  {nodes.map((n) => {
                    const p = points.get(n.node_id)!;
                    return (
                      <foreignObject
                        key={n.node_id}
                        x={p.x - 120}
                        y={p.y - 35}
                        width="240"
                        height="75"
                      >
                        <button
                          className={
                            "graph-node " +
                            (n.node_id === node ? "selected" : "")
                          }
                          style={{
                            opacity:
                              search &&
                              !n.name
                                .toLowerCase()
                                .includes(search.toLowerCase())
                                ? 0.3
                                : 1,
                          }}
                          onClick={() => setNode(n.node_id)}
                        >
                          <strong>{n.name}</strong>
                          <code>{n.version_label || "未命名版本"}</code>
                        </button>
                      </foreignObject>
                    );
                  })}
                </svg>
              </div>
              <div className="graph-list">
                {nodes
                  .filter((n) =>
                    n.name.toLowerCase().includes(search.toLowerCase()),
                  )
                  .map((n) => (
                    <button key={n.node_id} onClick={() => setNode(n.node_id)}>
                      {names.get(n.node_id)}
                    </button>
                  ))}
              </div>
              {selected && (
                <section className="node-detail">
                  <div className="row">
                    <div>
                      <strong>{selected.name}</strong>
                      <div>
                        <code>{selected.version_label || "未命名版本"}</code>
                        <details>
                          <summary>审计信息</summary>
                          <code>{selected.skill_version_id}</code>
                        </details>
                      </div>
                    </div>
                    <button
                      onClick={() =>
                        open({
                          source: "graph",
                          id: selected.skill_id,
                          name: selected.name,
                          version: selected.skill_version_id,
                        })
                      }
                    >
                      打开此版本
                    </button>
                  </div>
                </section>
              )}
              <details className="relations">
                <summary>关系 · {edges.length}</summary>
                {edges.map((e) => (
                  <p key={e.edge_id}>
                    {names.get(e.source_node_id)} →{" "}
                    {names.get(e.target_node_id)}{" "}
                    <span className="badge">{e.edge_type}</span>
                  </p>
                ))}
              </details>
            </>
          ) : (
            <Empty>
              {data.loading || graphs.loading || snapshots.loading
                ? "加载中…"
                : "当前工作区暂无管理图谱节点"}
            </Empty>
          )}
        </>
      )}
    </>
  );
}
