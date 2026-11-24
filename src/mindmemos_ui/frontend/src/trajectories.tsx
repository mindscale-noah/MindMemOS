import { useEffect, useId, useRef, useState } from "react";
import { api, query, type TrajectoryStatistics, type Version } from "./api";
import { Empty, Message, VersionOptions } from "./components";

type Range = [number, number];
const number = (value: number) => value.toLocaleString("en-US");
const dayLabel = (day: string) => day.slice(5).replace("-", "/");

export function Trajectories({
  active,
  skillId,
  version,
  versions,
  setVersion,
}: {
  active: boolean;
  skillId: string;
  version: string;
  versions: Version[];
  setVersion: (version: string) => void;
}) {
  const [result, setResult] = useState<{
    key: string;
    data: TrajectoryStatistics;
  }>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [range, setRange] = useState<Range>();
  const [hover, setHover] = useState<number>();
  const dragging = useRef<number>();
  const request = useRef(0);
  const pending = useRef("");
  const clipId = useId().replace(/:/g, "");
  const key = `${skillId}:${version}`;
  const data = result?.key === key ? result.data : undefined;

  async function load() {
    if (!version || pending.current === key) return;
    const sequence = ++request.current;
    pending.current = key;
    setBusy(true);
    setError("");
    try {
      const value = await api<TrajectoryStatistics>(
        "trajectory-statistics?" +
          query({ skill_id: skillId, version_id: version }),
      );
      if (sequence === request.current) {
        setResult({ key, data: value });
        setRange(undefined);
        setHover(undefined);
      }
    } catch (cause) {
      if (sequence === request.current) setError((cause as Error).message);
    } finally {
      if (sequence === request.current) {
        pending.current = "";
        setBusy(false);
      }
    }
  }
  // Retain this version's result across tab changes. A version change waits for Query.
  useEffect(() => {
    if (active && version && !result && !error) void load();
  }, [active, key]);
  useEffect(
    () => () => {
      request.current++;
    },
    [],
  );
  if (!active) return null;

  const days = data?.days || [];
  const peak = Math.max(1, ...days.map((day) => day.total));
  const ceiling = Math.ceil(peak / 4) * 4;
  const selected: Range =
    hover !== undefined ? [hover, hover] : range || [0, 29];
  const chosen = days.slice(selected[0], selected[1] + 1);
  const totals = chosen.reduce(
    (sum, day) => ({
      total: sum.total + day.total,
      consumed: sum.consumed + day.consumed,
      unconsumed: sum.unconsumed + day.unconsumed,
    }),
    { total: 0, consumed: 0, unconsumed: 0 },
  );
  const selectionLabel =
    chosen.length === 1
      ? chosen[0]?.day
      : `${chosen[0]?.day || ""} — ${chosen.at(-1)?.day || ""}`;
  const firstX = 54,
    step = 30,
    baseY = 270,
    height = 220;
  function indexAt(event: React.PointerEvent<SVGSVGElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    return Math.min(
      29,
      Math.max(
        0,
        Math.floor(
          (((event.clientX - bounds.left) / bounds.width) * 1000 - firstX) /
            step,
        ),
      ),
    );
  }

  return (
    <section className="trajectory-page" aria-label="Trajectory statistics">
      <div className="version-toolbar">
        <label className="version-picker">
          Version
          <select
            value={version}
            onChange={(event) => {
              setVersion(event.target.value);
              setError("");
            }}
          >
            <VersionOptions versions={versions} />
          </select>
        </label>
        <button
          className="primary"
          disabled={busy || !version}
          onClick={() => void load()}
        >
          {busy ? "Querying…" : "Query"}
        </button>
      </div>
      <Message text={error} error />
      {data ? (
        <>
          <div className="trajectory-metrics">
            <Metric
              label="Total trajectories"
              value={number(data.totals.total)}
            />
            <Metric
              label="Consumed"
              value={number(data.totals.consumed)}
              tone="consumed"
            />
            <Metric
              label="Not consumed"
              value={number(data.totals.unconsumed)}
              tone="unconsumed"
            />
            <Metric
              label="Last evolve"
              value={
                data.last_evolved_at
                  ? new Date(data.last_evolved_at).toLocaleString("en-GB", {
                      timeZone: data.timezone,
                      year: "numeric",
                      month: "2-digit",
                      day: "2-digit",
                      hour: "2-digit",
                      minute: "2-digit",
                      hour12: false,
                    })
                  : "—"
              }
            />
          </div>
          <div className="trajectory-chart-card">
            <div className="trajectory-chart-heading">
              <h2>Last 30 days</h2>
              <div className="trajectory-legend">
                <span className="consumed">Consumed</span>
                <span className="unconsumed">Not consumed</span>
              </div>
            </div>
            <svg
              className="trajectory-chart"
              viewBox="0 0 1000 322"
              aria-label={`Daily trajectories from ${days[0].day} to ${days[29].day}`}
              role="group"
              onPointerDown={(event) => {
                if (event.button !== 0) return;
                const index = indexAt(event);
                dragging.current = index;
                setRange([index, index]);
                setHover(undefined);
                event.currentTarget.setPointerCapture(event.pointerId);
              }}
              onPointerMove={(event) => {
                const index = indexAt(event);
                if (dragging.current !== undefined) {
                  setRange([
                    Math.min(index, dragging.current),
                    Math.max(index, dragging.current),
                  ]);
                  setHover(undefined);
                } else setHover(index);
              }}
              onPointerUp={(event) => {
                dragging.current = undefined;
                event.currentTarget.releasePointerCapture(event.pointerId);
              }}
              onPointerCancel={() => {
                dragging.current = undefined;
              }}
              onPointerLeave={() => setHover(undefined)}
            >
              {[0, 1, 2, 3, 4].map((tick) => (
                <g key={tick}>
                  <line
                    x1={firstX}
                    x2="954"
                    y1={baseY - (height * tick) / 4}
                    y2={baseY - (height * tick) / 4}
                    className="trajectory-grid"
                  />
                  <text
                    x="42"
                    y={baseY - (height * tick) / 4 + 5}
                    textAnchor="end"
                  >
                    {number((ceiling * tick) / 4)}
                  </text>
                </g>
              ))}
              {range && (
                <rect
                  className="trajectory-range"
                  x={firstX + range[0] * step}
                  y="40"
                  width={(range[1] - range[0] + 1) * step}
                  height="232"
                  rx="8"
                />
              )}
              {days.map((day, index) => {
                const x = firstX + index * step + 5,
                  totalHeight = (day.total / ceiling) * height,
                  consumedHeight = (day.consumed / ceiling) * height;
                const radius = Math.min(10, totalHeight / 2);
                const shape = `M ${x} ${baseY} V ${baseY - totalHeight + radius} Q ${x} ${baseY - totalHeight} ${x + radius} ${baseY - totalHeight} H ${x + 20 - radius} Q ${x + 20} ${baseY - totalHeight} ${x + 20} ${baseY - totalHeight + radius} V ${baseY} Z`;
                return (
                  <g
                    key={day.day}
                    role="button"
                    tabIndex={0}
                    aria-label={`${day.day}: ${day.total} total, ${day.consumed} consumed, ${day.unconsumed} not consumed`}
                    onFocus={() => setHover(index)}
                    onBlur={() => setHover(undefined)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setRange([index, index]);
                        setHover(undefined);
                      }
                      if (event.key === "Escape") {
                        setRange(undefined);
                        setHover(undefined);
                      }
                    }}
                  >
                    <defs>
                      <clipPath id={`${clipId}-${index}`}>
                        <path d={shape} />
                      </clipPath>
                    </defs>
                    <rect
                      x={x - 5}
                      y="40"
                      width={step}
                      height="232"
                      className="trajectory-hit"
                    />
                    <g clipPath={`url(#${clipId}-${index})`}>
                      <rect
                        x={x}
                        y={baseY - totalHeight}
                        width="20"
                        height={totalHeight}
                        className="trajectory-unconsumed"
                      />
                      <rect
                        x={x}
                        y={baseY - consumedHeight}
                        width="20"
                        height={consumedHeight}
                        className="trajectory-consumed"
                      />
                    </g>
                    {hover === index && (
                      <rect
                        x={x - 4}
                        y="40"
                        width="28"
                        height="232"
                        className="trajectory-hover"
                        rx="8"
                      />
                    )}
                    {[0, 5, 10, 15, 20, 25, 29].includes(index) && (
                      <text x={x + 10} y="306" textAnchor="middle">
                        {dayLabel(day.day)}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
            <div className="trajectory-selection-heading">
              <strong>{selectionLabel}</strong>
              {range && (
                <button
                  className="link"
                  onClick={() => {
                    setRange(undefined);
                    setHover(undefined);
                  }}
                >
                  Clear selection
                </button>
              )}
            </div>
            <div className="trajectory-selection" aria-live="polite">
              <Metric label="Total trajectories" value={number(totals.total)} />
              <Metric
                label="Consumed"
                value={number(totals.consumed)}
                tone="consumed"
              />
              <Metric
                label="Not consumed"
                value={number(totals.unconsumed)}
                tone="unconsumed"
              />
            </div>
          </div>
        </>
      ) : (
        <Empty>
          {busy ? "Loading…" : "Choose a version and query its trajectories"}
        </Empty>
      )}
    </section>
  );
}

function Metric({
  label,
  value,
  tone = "",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className={`trajectory-metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
