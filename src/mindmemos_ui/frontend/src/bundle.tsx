import { useState } from "react";
import { Empty } from "./components";
import { SplitDiff } from "./split-diff";
import { FilePreview } from "./file-preview";

export type BundleContent = {
  files: Record<string, string>;
  resources: string[];
  binary_files?: string[];
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
  added: "新增",
  deleted: "删除",
  modified: "修改",
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
              <small>{labels[changes.get(path)!.change] || "变化"}</small>
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
  beforeLabel = "基准",
  afterLabel = "目标",
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
  const changes = new Map(comparison?.files.map((file) => [file.path, file]));
  const [chosen, setChosen] = useState("");
  const selected = paths.includes(chosen)
    ? chosen
    : comparison?.files[0]?.path ||
      (paths.includes("SKILL.md") ? "SKILL.md" : paths[0]);
  const change = changes.get(selected);
  function notice(bundle: BundleContent) {
    const text = bundle.files[selected];
    if (text === "") return "空文件";
    if (text !== undefined) return undefined;
    return bundle.resources.includes(selected) ||
      bundle.binary_files?.includes(selected)
      ? "二进制文件，暂不支持文本预览"
      : "此版本中不存在该文件";
  }
  if (!paths.length) return <Empty>暂无文件</Empty>;
  return (
    <>
      {comparison && (
        <p className="muted">
          {changes.size
            ? `${changes.size} 个文件有差异，蓝色标记`
            : "所有文件均无差异"}
          {comparison.implementation_changed && "；实现配置有变化"}
        </p>
      )}
      <div className="bundle-browser">
        <div
          className="bundle-sidebar"
          role="navigation"
          aria-label="Skill 文件"
        >
          <div className="code-title">文件 · {paths.length}</div>
          <FileTree
            paths={paths}
            selected={selected}
            changes={changes}
            select={setChosen}
          />
        </div>
        <div className="bundle-content">
          <div className="code-title">
            {selected}
            {comparison && (
              <span className="badge">
                {change ? labels[change.change] || "变化" : "无差异"}
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
              <Empty>空文件</Empty>
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
            <Empty>二进制文件，暂不支持文本预览</Empty>
          )}
        </div>
      </div>
    </>
  );
}
