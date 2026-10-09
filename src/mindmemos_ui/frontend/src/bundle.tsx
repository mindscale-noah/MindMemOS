import { useState } from "react";
import { Empty } from "./components";
import { SplitDiff } from "./split-diff";
import { FilePreview } from "./file-preview";

export type BundleContent = {
  files: Record<string, string>;
  resources: string[];
  binary_files?: string[];
  raw_resources?: Record<string, string>;
};
export type FileChange = {
  path: string;
  change: string;
  binary: boolean;
  diff: string;
};
export type BundleComparison = {
  before: BundleContent;
  after: BundleContent;
  files: FileChange[];
  implementation_changed: boolean;
};
const labels: Record<string, string> = {
  added: "Added",
  deleted: "Deleted",
  modified: "Modified",
};

function FileTree({
  paths,
  selected,
  changes,
  select,
  prefix = "",
}: {
  paths: string[];
  selected: string;
  changes: Map<string, FileChange>;
  select: (path: string) => void;
  prefix?: string;
}) {
  const folders = new Map<string, string[]>();
  const files: string[] = [];
  for (const path of paths) {
    const tail = path.slice(prefix.length);
    const slash = tail.indexOf("/");
    if (slash < 0) files.push(path);
    else {
      const folder = tail.slice(0, slash);
      folders.set(folder, [...(folders.get(folder) || []), path]);
    }
  }
  return (
    <ul className="file-tree">
      {[...folders].map(([folder, children]) => (
        <li key={folder + "/"}>
          <details open>
            <summary
              className={
                children.some((path) => changes.has(path)) ? "changed" : ""
              }
            >
              {folder}/
            </summary>
            <FileTree
              paths={children}
              selected={selected}
              changes={changes}
              select={select}
              prefix={prefix + folder + "/"}
            />
          </details>
        </li>
      ))}
      {files.map((path) => (
        <li key={path}>
          <button
            className={`file-button ${changes.has(path) ? "changed" : ""} ${selected === path ? "selected" : ""}`}
            onClick={() => select(path)}
            aria-current={selected === path ? "true" : undefined}
            title={path}
          >
            <span>{path.slice(prefix.length)}</span>
            {changes.has(path) && (
              <small>{labels[changes.get(path)!.change] || "Changed"}</small>
            )}
          </button>
        </li>
      ))}
    </ul>
  );
}

export function BundleBrowser({
  content,
  comparison,
  beforeLabel = "Base",
  afterLabel = "Target",
}: {
  content: BundleContent;
  comparison?: BundleComparison;
  beforeLabel?: string;
  afterLabel?: string;
}) {
  const paths = [
    ...new Set([
      ...Object.keys(content.files),
      ...content.resources,
      ...Object.keys(comparison?.before.files || {}),
      ...(comparison?.before.resources || []),
      ...(comparison?.files.map((file) => file.path) || []),
    ]),
  ].sort();
  const [showFiles, setShowFiles] = useState(true);
  const changes = new Map(comparison?.files.map((file) => [file.path, file]));
  const [chosen, setChosen] = useState("");
  const selected = paths.includes(chosen)
    ? chosen
    : comparison?.files[0]?.path ||
      (paths.includes("SKILL.md") ? "SKILL.md" : paths[0]);
  const change = changes.get(selected);
  function notice(bundle: BundleContent) {
    const text = bundle.files[selected];
    if (text === "") return "Empty file";
    if (text !== undefined) return undefined;
    return bundle.resources.includes(selected) ||
      bundle.binary_files?.includes(selected)
      ? "Text previews are unavailable for binary files"
      : "This file does not exist in this version";
  }
  const sidebarToggle = (
    <button
      className="sidebar-toggle"
      aria-label={showFiles ? "Hide file sidebar" : "Show file sidebar"}
      title={showFiles ? "Hide file sidebar" : "Show file sidebar"}
      aria-expanded={showFiles}
      onClick={() => setShowFiles(!showFiles)}
    >
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        aria-hidden="true"
      >
        <rect x="3" y="4" width="18" height="16" rx="3" />
        <path d="M9 4v16" />
      </svg>
    </button>
  );
  if (!paths.length) return <Empty>No files</Empty>;
  return (
    <>
      {comparison && (
        <div className="bundle-toolbar">
          <p className="muted">
            {changes.size
              ? `${changes.size} changed ${changes.size === 1 ? "file" : "files"}, marked in blue`
              : "All files are identical"}
            {comparison.implementation_changed && "; implementation configuration has changed"}
          </p>
        </div>
      )}
      <div className={`bundle-browser ${showFiles ? "" : "files-collapsed"}`}>
        <div
          className="bundle-sidebar"
          hidden={!showFiles}
          role="navigation"
          aria-label="Skill files"
        >
          <div className="code-title file-tree-title">
            {sidebarToggle}
            <span>Files · {paths.length}</span>
          </div>
          <FileTree
            paths={paths}
            selected={selected}
            changes={changes}
            select={setChosen}
          />
        </div>
        <div className="bundle-content">
          <div className="code-title">
            {!showFiles && sidebarToggle}
            <span>{selected}</span>
            {comparison && (
              <span className="comparison-labels">
                {beforeLabel} → {afterLabel}
              </span>
            )}
            {comparison && (
              <span className="badge">
                {change ? labels[change.change] || "Changed" : "Unchanged"}
              </span>
            )}
          </div>
          {comparison ? (
            <SplitDiff
              path={selected}
              beforeLabel={beforeLabel}
              afterLabel={afterLabel}
              before={comparison.before.files[selected]}
              after={comparison.after.files[selected]}
              beforeNotice={notice(comparison.before)}
              afterNotice={notice(comparison.after)}
            />
          ) : content.files[selected] !== undefined ? (
            content.files[selected] === "" ? (
              <Empty>Empty file</Empty>
            ) : (
              <FilePreview
                key={selected}
                path={selected}
                text={content.files[selected]}
                paths={paths}
                onOpenFile={setChosen}
              />
            )
          ) : (
            <Empty>Text previews are unavailable for binary files</Empty>
          )}
        </div>
      </div>
    </>
  );
}
