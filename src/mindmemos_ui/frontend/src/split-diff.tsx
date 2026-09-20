import { Fragment } from "react";
import { diffLines } from "diff";

type Line = { number: number; text: string; changed?: boolean };
export type DiffRow = { left?: Line; right?: Line };

export function diffRows(before: string, after: string): DiffRow[] {
  const rows: DiffRow[] = [];
  let left = 1;
  let right = 1;
  let removed: Line[] = [];
  let added: Line[] = [];
  function flush() {
    for (let i = 0; i < Math.max(removed.length, added.length); i++) {
      rows.push({ left: removed[i], right: added[i] });
    }
    removed = [];
    added = [];
  }
  for (const part of diffLines(before, after)) {
    const lines = part.value.split("\n");
    if (lines[lines.length - 1] === "") lines.pop();
    if (!part.removed && !part.added) flush();
    for (const text of lines) {
      if (part.removed) removed.push({ number: left++, text, changed: true });
      else if (part.added) added.push({ number: right++, text, changed: true });
      else
        rows.push({
          left: { number: left++, text },
          right: { number: right++, text },
        });
    }
  }
  flush();
  return rows;
}

export function SplitDiff({
  path,
  beforeLabel,
  afterLabel,
  before,
  after,
  beforeNotice,
  afterNotice,
}: {
  path: string;
  beforeLabel: string;
  afterLabel: string;
  before?: string;
  after?: string;
  beforeNotice?: string;
  afterNotice?: string;
}) {
  const rows = diffRows(before || "", after || "");
  return (
    <div
      className="split-diff"
      role="region"
      aria-label={`${path} 左侧基准版本 ${beforeLabel}，右侧目标版本 ${afterLabel}`}
    >
      {(beforeNotice || afterNotice) && (
        <>
          <div className="split-notice">{beforeNotice}</div>
          <div className="split-notice">{afterNotice}</div>
        </>
      )}
      {rows.map((row, i) => (
        <Fragment key={i}>
          {(["left", "right"] as const).map((side) => {
            const line = row[side];
            return (
              <div
                key={side}
                className={`split-line ${!line ? "blank" : line.changed ? (side === "left" ? "removed" : "added") : ""}`}
              >
                <span className="line-number">{line?.number}</span>
                <span className="line-sign" aria-hidden="true">
                  {line?.changed ? (side === "left" ? "−" : "+") : ""}
                </span>
                <code>{line?.text || " "}</code>
              </div>
            );
          })}
        </Fragment>
      ))}
      <div className="split-notice">
        {before && !before.endsWith("\n") ? "文件末尾无换行" : ""}
      </div>
      <div className="split-notice">
        {after && !after.endsWith("\n") ? "文件末尾无换行" : ""}
      </div>
    </div>
  );
}
