import { lazy, Suspense, useState, useEffect, useMemo } from "react";
import { api, type Selection } from "./api";
import { useLoad, useRefresh, Message, Empty } from "./components";
import type { GraphNode, GraphEdge } from "./graph-flow";

const GraphCanvas = lazy(() =>
  import("./graph-flow").then((module) => ({ default: module.GraphCanvas })),
);
type GraphSelection = { kind: "node" | "edge"; id: string } | null;

export function Graphs({
  connected,
  open,
}: {
  connected: boolean;
  open: (s: Selection) => void;
}) {
  const [graph, setGraph] = useState("");
  const [snapshot, setSnapshot] = useState("");
  const [selection, setSelection] = useState<GraphSelection>(null);
  const [search, setSearch] = useState("");
  const [edgeType, setEdgeType] = useState("");
  const [reload, setReload] = useState(0);
  const refresh = useRefresh(() => setReload((current) => current + 1));
  const [downloadUrl, setDownloadUrl] = useState("");
  const graphs = useLoad<any[]>(
    () => (connected ? api("graphs") : Promise.resolve([])),
    "graphs" + connected + reload,
  );
  const snapshots = useLoad<any[]>(
    () =>
      connected && graph
        ? api(`graphs/${encodeURIComponent(graph)}/snapshots`)
        : Promise.resolve([]),
    connected + graph + reload,
  );
  const data = useLoad<{ nodes: GraphNode[]; edges: GraphEdge[] }>(
    () =>
      connected && graph && snapshot
        ? api(`graphs/${encodeURIComponent(graph)}/snapshots/${snapshot}`)
        : Promise.resolve({ nodes: [], edges: [] }),
    connected + graph + snapshot + reload,
  );
  useEffect(() => {
    if (
      !connected ||
      (graphs.value && !graphs.value.some((item) => item.graph_id === graph))
    ) {
      setGraph(connected ? graphs.value?.[0]?.graph_id || "" : "");
      setSnapshot("");
      setSelection(null);
    }
  }, [connected, graphs.value, graph]);
  useEffect(() => {
    const rows = snapshots.value;
    if (rows?.length) {
      const latest = [...rows].sort((a, b) =>
        b.created_at.localeCompare(a.created_at),
      )[0];
      const head = graphs.value?.find((item) => item.graph_id === graph)?.current_version_id;
      const current = rows.find((item) => item.graph_version_id === head);
      setSnapshot((current || latest).graph_version_id);
    }
  }, [snapshots.value, graphs.value, graph]);
  const nodes = connected && graph && snapshot ? data.value?.nodes || [] : [];
  const edges = useMemo(
    () =>
      (data.value?.edges || []).filter(
        (e) => !edgeType || e.edge_type === edgeType,
      ),
    [data.value?.edges, edgeType],
  );
  const relationTypes = useMemo(
    () =>
      [...new Set((data.value?.edges || []).map((e) => e.edge_type))].sort(),
    [data.value?.edges],
  );
  useEffect(() => {
    if (!connected || !graph || !snapshot || !data.value?.nodes.length) {
      setDownloadUrl("");
      return;
    }
    const name =
      graphs.value?.find((item) => item.graph_id === graph)?.name || graph;
    const payload = {
      format: "mindmemos.skill_graph.v1",
      graph: {
        graph_id: graph,
        name,
        graph_version_id: snapshot,
      },
      nodes: data.value.nodes,
      edges: data.value.edges,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2) + "\n"], {
        type: "application/json;charset=utf-8",
      }),
    );
    setDownloadUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [data.value, graph, snapshot, connected, graphs.value]);
  return (
    <>
      <div className="graph-selection-toolbar">
        <div className="fields">
          <label>
            Graph
            <select
              value={graph}
              onChange={(e) => {
                setGraph(e.target.value);
                setSnapshot("");
                setSelection(null);
                setSearch("");
                setEdgeType("");
              }}
            >
              <option value="">Select a graph</option>
              {graphs.value?.map((g) => (
                <option key={g.graph_id} value={g.graph_id}>
                  {g.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Graph snapshot
            <select
              value={snapshot}
              onChange={(e) => {
                setSnapshot(e.target.value);
                setSelection(null);
              }}
            >
              <option value="">Select a snapshot</option>
              {snapshots.value?.map((s) => (
                <option key={s.graph_version_id} value={s.graph_version_id}>
                  {s.status} · {s.graph_version_id.slice(0, 8)}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="graph-page-actions">
          <a
            className={`graph-export${!downloadUrl || data.loading ? " disabled" : ""}`}
            href={downloadUrl || undefined}
            download={`skill-graph-${snapshot.slice(0, 8)}.json`}
            aria-disabled={!downloadUrl || data.loading}
            onClick={(event) => {
              if (!downloadUrl || data.loading) event.preventDefault();
            }}
          >
            Export JSON
          </a>
          <button onClick={refresh.refresh}>Refresh</button>
        </div>
      </div>
      {!connected && <Message text="The Skill service is not connected." />}
      <Message
        text={refresh.error || graphs.error || snapshots.error || data.error}
        error
      />
      {nodes.length ? (
        <>
          <div className="toolbar">
            <input
              placeholder="Find Skill or Scene"
              aria-label="Find Skill or Scene"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <select
              aria-label="Relationship type"
              value={edgeType}
              onChange={(e) => {
                setEdgeType(e.target.value);
                setSelection(null);
              }}
            >
              <option value="">All relationships</option>
              {relationTypes.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </div>
          <Suspense
            fallback={
              <div className="graph-canvas graph-loading">Loading graph…</div>
            }
          >
            <GraphCanvas
              key={`${graph}:${snapshot}:${edgeType}:${reload}`}
              nodes={nodes}
              edges={edges}
              allEdges={data.value?.edges || []}
              relationTypes={relationTypes}
              search={search}
              selectedNodeId={selection?.kind === "node" ? selection.id : ""}
              selectedEdgeId={selection?.kind === "edge" ? selection.id : ""}
              onSelectNode={(id) => setSelection({ kind: "node", id })}
              onSelectEdge={(id) => setSelection({ kind: "edge", id })}
              onClose={() => setSelection(null)}
              onOpenVersion={(selected) =>
                open({
                  source: "graph",
                  id: selected.skill_id,
                  name: selected.skill_name || selected.name,
                  version: selected.skill_version_id,
                })
              }
            />
          </Suspense>
        </>
      ) : (
        <Empty>
          {data.loading || graphs.loading || snapshots.loading
            ? "Loading…"
            : !graphs.value?.length
              ? "No graphs"
              : !graph
                ? "Select a graph"
                : "This graph has no nodes"}
        </Empty>
      )}
    </>
  );
}
