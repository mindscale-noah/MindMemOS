import { useMemo, useState } from "react";
import dagre from "@dagrejs/dagre";
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  type NodeProps,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api } from "./api";
import { Message } from "./components";
import "./algorithm-visualization.css";

type Checkpoint = {
  id: string;
  tree_id: number;
  parent_id: string | null;
  depth: number;
  status: string;
  expanded: boolean;
  skill: { name: string; content: string };
};
type ForestNode = {
  winner: "tree" | "final" | null;
  checkpoint: Checkpoint;
  completed: number;
  total: number | null;
  score: number | null;
  gain: number | null;
  errors: number;
};
type TrainingProgress = {
  stage: "root" | "search" | "select" | "completed" | "unknown";
  roots_completed: number;
  roots_total: number | null;
  expansions_completed: number;
  expansion_budget: number | null;
};
type Report = {
  runs: { path: string; nodes: ForestNode[]; progress: TrainingProgress }[];
  warnings: string[];
};
type FlowNode = Node<{ item: ForestNode }, "forest">;
const percent = (value: number | null) =>
  value === null ? "—" : `${(value * 100).toFixed(2)}%`;
const gain = (value: number | null) =>
  value === null
    ? "—"
    : `${value > 0 ? "+" : ""}${(value * 100).toFixed(2)} pp`;
const status = (node: ForestNode) =>
  node.checkpoint.status === "generated"
    ? "Pending validation"
    : node.checkpoint.status;
function Progress({ item }: { item: ForestNode }) {
  return (
    <div className="forest-progress">
      <span>
        Validation{" "}
        <b>
          {item.completed} / {item.total ?? "?"}
        </b>
      </span>
      <progress
        max={item.total || Math.max(item.completed, 1)}
        value={item.completed}
      />
    </div>
  );
}
function ForestCard({ data, selected }: NodeProps<FlowNode>) {
  const item = data.item;
  return (
    <div
      className={`forest-node ${item.winner ? `winner-${item.winner}` : ""} ${selected ? "selected" : ""}`}
    >
      <Handle type="target" position={Position.Top} />
      <div className="forest-node-heading">
        <strong>{item.checkpoint.id}</strong>
        <div className="forest-node-badges">
          <span className={`forest-badge ${item.checkpoint.status}`}>
            {status(item)}
          </span>
          {item.winner && (
            <span className="forest-badge winner">
              {item.winner === "final" ? "Final winner" : "Tree winner"}
            </span>
          )}
        </div>
      </div>
      <div className="forest-node-metrics">
        <div>
          <small>
            {item.total !== null && item.completed === item.total
              ? "Score"
              : "Partial score"}
          </small>
          <strong>{percent(item.score)}</strong>
        </div>
        <div>
          <small>Gain</small>
          <strong>{gain(item.gain)}</strong>
        </div>
      </div>
      <Progress item={item} />
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
const nodeTypes = { forest: ForestCard };

function TrainingStages({ progress }: { progress: TrainingProgress }) {
  const stages = ["root", "search", "select", "completed"] as const;
  const current = stages.findIndex((stage) => stage === progress.stage);
  const count =
    progress.stage === "root"
      ? progress.roots_completed
      : progress.expansions_completed;
  const total =
    progress.stage === "root"
      ? progress.roots_total
      : progress.expansion_budget;
  return (
    <div className="forest-training-progress" aria-label="Training progress">
      <span
        className="forest-stage-label"
        title="Latest stage evidenced by saved files; not a process health check."
      >
        Observed stage
      </span>
      <ol>
        {stages.map((stage, index) => (
          <li
            key={stage}
            className={
              index === current ? "current" : index < current ? "done" : ""
            }
            aria-current={index === current ? "step" : undefined}
          >
            <span>{index < current ? "✓" : index + 1}</span>
            {stage === "root"
              ? "Root"
              : stage === "search"
                ? "Search"
                : stage === "select"
                  ? "Select"
                  : "Completed"}
          </li>
        ))}
      </ol>
      <div className="forest-stage-detail">
        {progress.stage === "root" || progress.stage === "search" ? (
          <>
            <span>
              {count} / {total ?? "?"}{" "}
              {progress.stage === "root"
                ? "roots saved"
                : "expansion budget used"}
            </span>
            {total !== null && total > 0 && (
              <progress
                aria-label={
                  progress.stage === "root"
                    ? "Root checkpoints"
                    : "Search expansion budget"
                }
                max={total}
                value={count}
              />
            )}
          </>
        ) : (
          <span>
            {progress.stage === "select"
              ? "Evaluating / combining tree winners"
              : progress.stage === "completed"
                ? "Training result saved"
                : "Stage unavailable"}
          </span>
        )}
      </div>
    </div>
  );
}

export function AlgorithmVisualization() {
  const [path, setPath] = useState("");
  const [loadedPath, setLoadedPath] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [run, setRun] = useState(0);
  const [tree, setTree] = useState<number | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function load(directory: string, refresh = false) {
    setBusy(true);
    setError("");
    try {
      const result = await api<Report>(
        "algorithm-visualization/report",
        "POST",
        { path: directory },
      );
      setReport(result);
      setLoadedPath(directory);
      if (!refresh) {
        setRun(0);
        setTree(null);
        setSelected(null);
        setQuery("");
        setFilter("all");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load directory");
    } finally {
      setBusy(false);
    }
  }
  const nodes = report?.runs[run]?.nodes || [];
  const trees = [...new Set(nodes.map((item) => item.checkpoint.tree_id))].sort(
    (a, b) => a - b,
  );
  const currentTree = tree !== null && trees.includes(tree) ? tree : trees[0];
  const treeNodes = nodes.filter(
    (item) => item.checkpoint.tree_id === currentTree,
  );
  const active =
    treeNodes.find((item) => item.checkpoint.id === selected) || treeNodes[0];
  const parent = nodes.find(
    (item) => item.checkpoint.id === active?.checkpoint.parent_id,
  );
  const graph = useMemo(() => {
    const layout = new dagre.graphlib.Graph()
      .setGraph({ rankdir: "TB", nodesep: 36, ranksep: 65 })
      .setDefaultEdgeLabel(() => ({}));
    const ids = new Set(treeNodes.map((item) => item.checkpoint.id));
    treeNodes.forEach((item) =>
      layout.setNode(item.checkpoint.id, { width: 250, height: 158 }),
    );
    const edges = treeNodes
      .filter(
        (item) =>
          item.checkpoint.parent_id && ids.has(item.checkpoint.parent_id),
      )
      .map((item) => ({
        id: item.checkpoint.id,
        source: item.checkpoint.parent_id!,
        target: item.checkpoint.id,
        style: { stroke: "#d7a4aa", strokeWidth: 1.5 },
        type: "smoothstep",
      }));
    edges.forEach((edge) => layout.setEdge(edge.source, edge.target));
    dagre.layout(layout);
    const flowNodes: FlowNode[] = treeNodes.map((item) => {
      const point = layout.node(item.checkpoint.id);
      const matches =
        (filter === "all" || item.checkpoint.status === filter) &&
        item.checkpoint.id.toLowerCase().includes(query.toLowerCase());
      return {
        id: item.checkpoint.id,
        type: "forest",
        data: { item },
        position: { x: point.x - 125, y: point.y - 79 },
        selected: item === active,
        style: { opacity: matches ? 1 : 0.22 },
      };
    });
    return { nodes: flowNodes, edges };
  }, [treeNodes, active, filter, query]);
  return (
    <section className="algorithm-page">
      <form
        className="forest-source"
        onSubmit={(event) => {
          event.preventDefault();
          void load(path, path === loadedPath);
        }}
      >
        <select aria-label="Algorithm type" defaultValue="skill-forest">
          <option value="skill-forest">skill-forest</option>
        </select>
        <input
          id="forest-path"
          aria-label="Output directory"
          value={path}
          onChange={(event) => setPath(event.target.value)}
          placeholder="/path/to/experiment/output"
          required
        />
        <button
          className="primary"
          disabled={busy || !path.trim()}
          aria-busy={busy}
        >
          Visualize
        </button>
      </form>
      <Message
        error
        text={
          error
            ? error + (report ? " — Previous snapshot remains visible." : "")
            : ""
        }
      />
      {report?.warnings.length ? (
        <details className="forest-warning">
          <summary>{report.warnings.length} evidence notice(s)</summary>
          {report.warnings.map((warning, i) => (
            <p key={i}>{warning}</p>
          ))}
        </details>
      ) : null}
      {report && (
        <>
          {report.runs.length > 1 && (
            <label>
              Training run{" "}
              <select
                value={run}
                onChange={(e) => {
                  setRun(Number(e.target.value));
                  setTree(null);
                  setSelected(null);
                }}
              >
                {report.runs.map((item, i) => (
                  <option key={item.path} value={i}>
                    {item.path}
                  </option>
                ))}
              </select>
            </label>
          )}
          <div className="forest-stats">
            {[
              ["Trees", trees.length],
              ["Nodes", nodes.length],
              [
                "Accepted",
                nodes.filter((n) => n.checkpoint.status === "accepted").length,
              ],
              [
                "Rejected",
                nodes.filter((n) => n.checkpoint.status === "rejected").length,
              ],
              [
                "Pending validation",
                nodes.filter((n) => n.checkpoint.status === "generated").length,
              ],
            ].map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
          {report.runs[run]?.progress && (
            <TrainingStages progress={report.runs[run].progress} />
          )}
          {nodes.length ? (
            <div className="forest-explorer">
              <div className="forest-trees">
                <h3>
                  Forest <small>{trees.length} trees</small>
                </h3>
                {trees.map((id) => {
                  const items = nodes.filter(
                    (n) => n.checkpoint.tree_id === id,
                  );
                  return (
                    <button
                      className={id === currentTree ? "active" : ""}
                      key={id}
                      onClick={() => {
                        setTree(id);
                        setSelected(null);
                      }}
                    >
                      <strong>Tree {id}</strong>
                      <span>
                        {items.length} nodes ·{" "}
                        {
                          items.filter(
                            (n) => n.checkpoint.status === "accepted",
                          ).length
                        }{" "}
                        accepted
                      </span>
                    </button>
                  );
                })}
              </div>
              <div className="forest-canvas">
                <div className="forest-toolbar">
                  <strong>Tree {currentTree}</strong>
                  <input
                    aria-label="Find node"
                    placeholder="Find node…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                  <select
                    aria-label="Highlight status"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                  >
                    {["all", "root", "accepted", "rejected", "generated"].map(
                      (value) => (
                        <option key={value} value={value}>
                          {value === "all"
                            ? "All statuses"
                            : value === "generated"
                              ? "Pending validation"
                              : value}
                        </option>
                      ),
                    )}
                  </select>
                </div>
                <ReactFlow
                  key={`${run}-${currentTree}`}
                  nodes={graph.nodes}
                  edges={graph.edges}
                  nodeTypes={nodeTypes}
                  fitView
                  minZoom={0.15}
                  maxZoom={1.6}
                  nodesDraggable={false}
                  nodesConnectable={false}
                  onNodeClick={(_, node) => setSelected(node.id)}
                >
                  <Background color="#e9dadd" gap={20} />
                  <Controls showInteractive={false} />
                </ReactFlow>
              </div>
              <div className="forest-inspector">
                {active && (
                  <>
                    <div className="forest-inspector-title">
                      <small>NODE DETAILS</small>
                      <span
                        className={`forest-badge ${active.checkpoint.status}`}
                      >
                        {status(active)}
                      </span>
                    </div>
                    <h3>{active.checkpoint.id}</h3>
                    <div className="forest-detail-gain">
                      {gain(active.gain)}
                      <small>Validation gain</small>
                    </div>
                    <Progress item={active} />
                    <dl>
                      <dt>Observed score</dt>
                      <dd>{percent(active.score)}</dd>
                      <dt>Parent score</dt>
                      <dd>{percent(parent?.score ?? null)}</dd>
                      <dt>Depth</dt>
                      <dd>{active.checkpoint.depth}</dd>
                      <dt>Expansion</dt>
                      <dd>
                        {active.checkpoint.expanded
                          ? "Completed"
                          : "Not completed"}
                      </dd>
                      <dt>Invalid records</dt>
                      <dd>{active.errors}</dd>
                    </dl>
                    {active.completed !== active.total && (
                      <p className="forest-note">
                        Partial validation. Observed score is not a full-set
                        result; final gain is unavailable.
                      </p>
                    )}
                    <details>
                      <summary>Skill content</summary>
                      <pre>
                        {active.checkpoint.skill.content ||
                          "No Skill content saved."}
                      </pre>
                    </details>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="forest-empty">
              No readable nodes yet. Refresh after checkpoints are saved.
            </div>
          )}
        </>
      )}
      {!report && (
        <div className="forest-empty">
          <span className="forest-empty-icon">⑂</span>
          <h3>Your forest, at a glance</h3>
          <p>
            Enter an output directory to explore branches, compare gains,
            <br />
            and inspect accepted and rejected candidates.
          </p>
        </div>
      )}
    </section>
  );
}
