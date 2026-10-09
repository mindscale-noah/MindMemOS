import { useEffect, useMemo, useRef, useState } from "react";
import dagre from "@dagrejs/dagre";
import {
  ReactFlow,
  Background,
  BaseEdge,
  Controls,
  EdgeToolbar,
  Handle,
  MarkerType,
  MiniMap,
  NodeToolbar,
  Position,
  ReactFlowProvider,
  applyNodeChanges,
  getSmoothStepPath,
  useNodesInitialized,
  useReactFlow,
  useViewport,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeChange,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

export type GraphNode = {
  node_id: string;
  skill_id: string;
  skill_version_id: string;
  name: string;
  version_label?: string;
};

export type GraphEdge = {
  edge_id: string;
  source_node_id: string;
  target_node_id: string;
  edge_type: string;
};

const NODE_WIDTH = 244;
const NODE_HEIGHT = 112;
const EDGE_COLORS = ["#497b9a", "#9171b2", "#b9784a", "#438675", "#bd657b"];
const POPOVER_WIDTH = 292;
const POPOVER_HEIGHT = 230;

type SkillNode = Node<
  {
    record: GraphNode;
    relationCount: number;
    dimmed: boolean;
    popoverPosition: Position;
    popoverAlign: "start" | "center" | "end";
    onSelect: () => void;
    onClose: () => void;
    onOpenVersion: () => void;
  },
  "skill"
>;

function SkillCard({ data, selected }: NodeProps<SkillNode>) {
  const version = data.record.version_label || "Unnamed version";
  return (
    <>
      {selected && (
        <NodeToolbar
          isVisible
          position={data.popoverPosition}
          align={data.popoverAlign}
          offset={12}
          className="graph-popover nodrag nopan"
          role="dialog"
          aria-label={`${data.record.name} Details`}
          onClick={(event) => event.stopPropagation()}
        >
          <div className="graph-popover-heading">
            <div>
              <span className="graph-popover-kicker">SKILL DETAILS</span>
              <strong>{data.record.name}</strong>
            </div>
            <button
              type="button"
              className="graph-popover-close"
              aria-label="Close Skill details"
              onClick={data.onClose}
            >
              ×
            </button>
          </div>
          <dl className="graph-popover-fields">
            <dt>Version</dt>
            <dd>
              <code>{version}</code>
            </dd>
            <dt>Relationships</dt>
            <dd>{data.relationCount} relationships</dd>
            <>
              <dt>Skill ID</dt>
              <dd>
                <code>{data.record.skill_id}</code>
              </dd>
              <dt>Version ID</dt>
              <dd>
                <code>{data.record.skill_version_id}</code>
              </dd>
            </>
          </dl>
          <button
            type="button"
            className="primary graph-popover-action"
            onClick={data.onOpenVersion}
          >
            Open this version
          </button>
        </NodeToolbar>
      )}
      <Handle type="target" position={Position.Top} isConnectable={false} />
      <button
        type="button"
        className={`graph-node${selected ? " selected" : ""}${data.dimmed ? " dimmed" : ""}`}
        onClick={data.onSelect}
        title={`${data.record.name} · ${version}`}
      >
        <span className="graph-node-kicker">
          SKILL <span>· {data.relationCount} relationships</span>
        </span>
        <strong>{data.record.name}</strong>
        <code>{version}</code>
      </button>
      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </>
  );
}

const nodeTypes = { skill: SkillCard };

type SkillEdge = Edge<
  {
    record: GraphEdge;
    sourceName: string;
    targetName: string;
    active: boolean;
    canvasWidth: number;
    canvasHeight: number;
    onClose: () => void;
  },
  "relation"
>;

function SkillRelation(props: EdgeProps<SkillEdge>) {
  const [path, labelX, labelY] = getSmoothStepPath(props);
  const viewport = useViewport();
  const data = props.data!;
  const screenX = labelX * viewport.zoom + viewport.x;
  const screenY = labelY * viewport.zoom + viewport.y;
  const alignX =
    data.canvasWidth - screenX >= POPOVER_WIDTH + 24 ? "left" : "right";
  const alignY = data.canvasHeight - screenY >= 170 ? "top" : "bottom";
  const offsetX = (alignX === "left" ? 14 : -14) / viewport.zoom;
  const offsetY = (alignY === "top" ? 12 : -12) / viewport.zoom;
  return (
    <>
      <BaseEdge
        id={props.id}
        path={path}
        style={props.style}
        markerEnd={props.markerEnd}
        interactionWidth={24}
      />
      {data.active && (
        <EdgeToolbar
          edgeId={props.id}
          x={labelX + offsetX}
          y={labelY + offsetY}
          alignX={alignX}
          alignY={alignY}
          isVisible
          className="graph-popover nodrag nopan"
          role="dialog"
          aria-label={`${data.record.edge_type} Relationship details`}
          onClick={(event) => event.stopPropagation()}
        >
          <div className="graph-popover-heading">
            <div>
              <span className="graph-popover-kicker">Relationship details</span>
              <strong>{data.record.edge_type}</strong>
            </div>
            <button
              type="button"
              className="graph-popover-close"
              aria-label="Close relationship details"
              onClick={data.onClose}
            >
              ×
            </button>
          </div>
          <div className="graph-relation-endpoints">
            <div>
              <span>Source Skill</span>
              <strong>{data.sourceName}</strong>
            </div>
            <span aria-hidden="true">→</span>
            <div>
              <span>Target Skill</span>
              <strong>{data.targetName}</strong>
            </div>
          </div>
          <p className="graph-popover-id">
            Relationship ID <code>{data.record.edge_id}</code>
          </p>
        </EdgeToolbar>
      )}
    </>
  );
}

const edgeTypes = { relation: SkillRelation };

function layoutNodes(nodes: GraphNode[], edges: GraphEdge[]): SkillNode[] {
  const layout = new dagre.graphlib.Graph();
  layout.setGraph({
    rankdir: "TB",
    nodesep: 54,
    ranksep: 94,
    marginx: 36,
    marginy: 36,
  });
  layout.setDefaultEdgeLabel(() => ({}));
  const ids = new Set(nodes.map((node) => node.node_id));
  nodes.forEach((node) =>
    layout.setNode(node.node_id, { width: NODE_WIDTH, height: NODE_HEIGHT }),
  );
  edges.forEach((edge) => {
    if (ids.has(edge.source_node_id) && ids.has(edge.target_node_id)) {
      layout.setEdge(edge.source_node_id, edge.target_node_id);
    }
  });
  dagre.layout(layout);
  return nodes.map((node) => {
    const { x, y } = layout.node(node.node_id);
    return {
      id: node.node_id,
      type: "skill",
      position: { x: x - NODE_WIDTH / 2, y: y - NODE_HEIGHT / 2 },
      data: {
        record: node,
        relationCount: 0,
        dimmed: false,
        popoverPosition: Position.Right,
        popoverAlign: "center",
        onSelect: () => {},
        onClose: () => {},
        onOpenVersion: () => {},
      },
    };
  });
}

function Flow({
  nodes,
  edges,
  allEdges,
  relationTypes,
  search,
  selectedNodeId,
  selectedEdgeId,
  onSelectNode,
  onSelectEdge,
  onClose,
  onOpenVersion,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  allEdges: GraphEdge[];
  relationTypes: string[];
  search: string;
  selectedNodeId: string;
  selectedEdgeId: string;
  onSelectNode: (id: string) => void;
  onSelectEdge: (id: string) => void;
  onClose: () => void;
  onOpenVersion: (node: GraphNode) => void;
}) {
  const [positions, setPositions] = useState(() => layoutNodes(nodes, edges));
  const canvas = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const flow = useReactFlow<SkillNode, SkillEdge>();
  const viewport = useViewport();
  const initialized = useNodesInitialized();
  const term = search.trim().toLocaleLowerCase();
  const counts = useMemo(() => {
    const result = new Map(nodes.map((node) => [node.node_id, 0]));
    allEdges.forEach((edge) => {
      if (result.has(edge.source_node_id))
        result.set(edge.source_node_id, result.get(edge.source_node_id)! + 1);
      if (result.has(edge.target_node_id))
        result.set(edge.target_node_id, result.get(edge.target_node_id)! + 1);
    });
    return result;
  }, [nodes, allEdges]);
  const visibleTypes = useMemo(
    () => [...new Set(edges.map((edge) => edge.edge_type))].sort(),
    [edges],
  );
  const colors = new Map(
    relationTypes.map((type, index) => [
      type,
      EDGE_COLORS[index % EDGE_COLORS.length],
    ]),
  );
  const nodeById = new Map(nodes.map((node) => [node.node_id, node]));
  const flowNodes: SkillNode[] = positions.map((position) => {
    const node = nodeById.get(position.id)!;
    const left = position.position.x * viewport.zoom + viewport.x;
    const right = left + NODE_WIDTH * viewport.zoom;
    const bottom =
      position.position.y * viewport.zoom +
      viewport.y +
      NODE_HEIGHT * viewport.zoom;
    const popoverPosition =
      canvasSize.width - right >= POPOVER_WIDTH + 20
        ? Position.Right
        : left >= POPOVER_WIDTH + 20
          ? Position.Left
          : canvasSize.height - bottom >= POPOVER_HEIGHT + 20
            ? Position.Bottom
            : Position.Top;
    const center = left + (NODE_WIDTH * viewport.zoom) / 2;
    const popoverAlign =
      center < POPOVER_WIDTH / 2 + 12
        ? "start"
        : center > canvasSize.width - POPOVER_WIDTH / 2 - 12
          ? "end"
          : "center";
    return {
      ...position,
      selected: node.node_id === selectedNodeId,
      data: {
        record: node,
        relationCount: counts.get(node.node_id) || 0,
        dimmed: !!term && !node.name.toLocaleLowerCase().includes(term),
        popoverPosition,
        popoverAlign,
        onSelect: () => onSelectNode(node.node_id),
        onClose,
        onOpenVersion: () => onOpenVersion(node),
      },
    };
  });
  const flowEdges: SkillEdge[] = edges.map((edge) => {
    const active = edge.edge_id === selectedEdgeId;
    const relatedToNode =
      !!selectedNodeId &&
      (edge.source_node_id === selectedNodeId ||
        edge.target_node_id === selectedNodeId);
    const color = colors.get(edge.edge_type)!;
    return {
      id: edge.edge_id,
      source: edge.source_node_id,
      target: edge.target_node_id,
      type: "relation",
      ariaLabel: `${edge.edge_type}：${nodeById.get(edge.source_node_id)?.name || edge.source_node_id} → ${nodeById.get(edge.target_node_id)?.name || edge.target_node_id}`,
      style: {
        stroke: color,
        strokeWidth: active || relatedToNode ? 2.8 : 1.8,
        opacity:
          (selectedNodeId || selectedEdgeId) && !active && !relatedToNode
            ? 0.35
            : 0.8,
      },
      markerEnd: { type: MarkerType.ArrowClosed, color, width: 16, height: 16 },
      interactionWidth: 24,
      data: {
        record: edge,
        sourceName:
          nodeById.get(edge.source_node_id)?.name || edge.source_node_id,
        targetName:
          nodeById.get(edge.target_node_id)?.name || edge.target_node_id,
        active,
        canvasWidth: canvasSize.width,
        canvasHeight: canvasSize.height,
        onClose,
      },
    };
  });

  useEffect(() => {
    if (!initialized || !canvas.current) return;
    const focus = () => {
      if (!canvas.current?.clientWidth) return;
      const width = canvas.current.clientWidth;
      const height = canvas.current.clientHeight;
      setCanvasSize((current) =>
        current.width === width && current.height === height
          ? current
          : { width, height },
      );
      if (term) {
        const match = nodes.find((node) =>
          node.name.toLocaleLowerCase().includes(term),
        );
        if (match)
          void flow.fitView({
            nodes: [{ id: match.node_id }],
            duration: 250,
            maxZoom: 1.15,
            padding: 0.8,
          });
      } else if (nodes.length > 5) {
        const anchor =
          positions.find((node) =>
            edges.some((edge) => edge.source_node_id === node.id),
          ) || positions[0];
        const zoom = 0.88;
        void flow.setViewport(
          {
            x:
              canvas.current.clientWidth / 2 -
              (anchor.position.x + NODE_WIDTH / 2) * zoom,
            y: 80 - anchor.position.y * zoom,
            zoom,
          },
          { duration: 250 },
        );
      } else {
        void flow.fitView({ padding: 0.18, maxZoom: 1.1 });
      }
    };
    const observer = new ResizeObserver(focus);
    observer.observe(canvas.current);
    focus();
    return () => observer.disconnect();
  }, [flow, initialized, nodes, edges, term]);

  const onNodesChange = (changes: NodeChange<SkillNode>[]) => {
    setPositions((current) =>
      applyNodeChanges(
        changes.filter(
          (change) =>
            change.type === "position" || change.type === "dimensions",
        ),
        current,
      ),
    );
  };

  return (
    <div
      ref={canvas}
      className="graph-canvas"
      role="region"
      aria-label="Skill relationship graph"
    >
      <div className="graph-legend" aria-label="Relationship color legend">
        {visibleTypes.map((type) => (
          <span key={type}>
            <i style={{ background: colors.get(type) }} />
            {type}
          </span>
        ))}
      </div>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgeClick={(_, edge) => onSelectEdge(edge.id)}
        onPaneClick={onClose}
        fitView={nodes.length <= 5}
        fitViewOptions={{ padding: 0.18, maxZoom: 1.1 }}
        minZoom={0.2}
        maxZoom={2}
        nodesConnectable={false}
        deleteKeyCode={null}
        edgesFocusable
        edgesReconnectable={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#e6ebf0" gap={24} size={1} />
        <Controls showInteractive={false} />
        {nodes.length > 12 && <MiniMap pannable zoomable nodeColor="#f0d9dd" />}
      </ReactFlow>
    </div>
  );
}

export function GraphCanvas(props: Parameters<typeof Flow>[0]) {
  return (
    <ReactFlowProvider>
      <Flow {...props} />
    </ReactFlowProvider>
  );
}
