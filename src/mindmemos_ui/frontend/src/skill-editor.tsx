import { useEffect, useState, type FormEvent } from "react";
import { api, query, versionPattern, type Ref } from "./api";
import { Dialog, Empty, Message, useLoad } from "./components";
import { type BundleContent, type FileChange } from "./bundle";
import { FilePreview } from "./file-preview";
import { SplitDiff } from "./split-diff";

function encodeText(text: string) {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function EditorTree({
  paths,
  folders,
  selected,
  changes,
  select,
  remove,
  undo,
  prefix = "",
}: {
  paths: string[];
  folders: string[];
  selected: string;
  changes: Map<string, FileChange>;
  select: (path: string) => void;
  remove: (path: string, folder?: boolean) => void;
  undo: (path: string) => void;
  prefix?: string;
}) {
  const children = [
    ...new Set(
      [...paths, ...folders.map((path) => path + "/")]
        .filter((path) => path.startsWith(prefix))
        .map((path) => path.slice(prefix.length))
        .filter(Boolean),
    ),
  ];
  const directories = [
    ...new Set(
      children
        .filter((path) => path.includes("/"))
        .map((path) => path.split("/")[0]),
    ),
  ].sort();
  return (
    <ul className="file-tree editor-tree">
      {children
        .filter((path) => !path.includes("/"))
        .sort()
        .map((name) => {
          const path = prefix + name;
          const change = changes.get(path)?.change;
          return (
            <li
              key={path}
              className={`editor-file-row ${change === "deleted" ? "deleted" : ""}`}
            >
              <button
                type="button"
                className={`file-button ${selected === path ? "selected" : ""}`}
                onClick={() => select(path)}
                title={path}
                aria-current={selected === path ? "true" : undefined}
              >
                <span>{name}</span>
                {change && (
                  <span
                    className={`file-status ${change}`}
                    title={change}
                    aria-label={change}
                  >
                    ●
                  </span>
                )}
              </button>
              {change === "deleted" ? (
                <button
                  type="button"
                  className="file-action"
                  aria-label={`Restore ${path}`}
                  title="Restore file"
                  onClick={() => undo(path)}
                >
                  ↶
                </button>
              ) : (
                path !== "SKILL.md" && (
                  <button
                    type="button"
                    className="file-action"
                    aria-label={`Delete ${path}`}
                    title="Delete file"
                    onClick={() => remove(path)}
                  >
                    ×
                  </button>
                )
              )}
            </li>
          );
        })}
      {directories.map((name) => {
        const path = prefix + name;
        return (
          <li key={path}>
            <details open>
              <summary>
                <span>{name}/</span>
                <button
                  type="button"
                  className="file-action"
                  aria-label={`Delete folder ${path}`}
                  title="Delete folder"
                  onClick={(event) => {
                    event.preventDefault();
                    remove(path, true);
                  }}
                >
                  ×
                </button>
              </summary>
              <EditorTree
                paths={paths}
                folders={folders}
                selected={selected}
                changes={changes}
                select={select}
                remove={remove}
                undo={undo}
                prefix={path + "/"}
              />
            </details>
          </li>
        );
      })}
    </ul>
  );
}

export function SkillEditor({
  version,
  label,
  dirty,
  close,
  done,
}: {
  version: Ref;
  label: string;
  dirty: (value: boolean) => void;
  close: () => void;
  done: (id: string) => void;
}) {
  const data = useLoad<BundleContent>(
    () => api("content?" + query(version)),
    JSON.stringify(version),
  );
  return (
    <>
      <Message text={data.error} error />
      {data.loading ? (
        <Empty>Loading editor…</Empty>
      ) : (
        data.value && (
          <EditorWorkspace
            key={JSON.stringify(version)}
            original={data.value}
            version={version}
            label={label}
            dirty={dirty}
            close={close}
            done={done}
          />
        )
      )}
    </>
  );
}

function EditorWorkspace({
  original,
  version,
  label,
  dirty,
  close,
  done,
}: {
  original: BundleContent;
  version: Ref;
  label: string;
  dirty: (value: boolean) => void;
  close: () => void;
  done: (id: string) => void;
}) {
  const [files, setFiles] = useState({ ...original.files });
  const [deletedDrafts, setDeletedDrafts] = useState<Record<string, string>>(
    {},
  );
  const [binary, setBinary] = useState([...(original.binary_files || [])]);
  const [folders, setFolders] = useState<string[]>([]);
  const [selected, setSelected] = useState("SKILL.md");
  const [view, setView] = useState("source");
  const [showFiles, setShowFiles] = useState(true);
  const [review, setReview] = useState(false);
  const [discarding, setDiscarding] = useState(false);
  const [creating, setCreating] = useState<"file" | "folder">();
  const [name, setName] = useState("");
  const [nameError, setNameError] = useState("");
  const [removing, setRemoving] = useState<{ path: string; folder: boolean }>();
  const [mode, setMode] = useState("patch");
  const [manual, setManual] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const livePaths = [...Object.keys(files), ...binary].sort();
  const originalPaths = [
    ...Object.keys(original.files),
    ...(original.binary_files || []),
  ];
  const paths = [...new Set([...livePaths, ...originalPaths])].sort();
  const changes = new Map<string, FileChange>();
  for (const path of paths) {
    const change = !livePaths.includes(path)
      ? "deleted"
      : !originalPaths.includes(path)
        ? "added"
        : files[path] !== original.files[path]
          ? "modified"
          : "";
    if (change)
      changes.set(path, {
        path,
        change,
        binary: original.binary_files?.includes(path) || false,
        diff: "",
      });
  }
  const changed = changes.size > 0;
  useEffect(() => {
    dirty(changed || busy);
    return () => dirty(false);
  }, [changed, busy, dirty]);
  const deleted = !livePaths.includes(selected);
  const isBinary =
    binary.includes(selected) ||
    (deleted && original.binary_files?.includes(selected));
  const markdown = /\.(md|markdown)$/i.test(selected);
  function cancel() {
    if (busy) return;
    if (changed) setDiscarding(true);
    else close();
  }
  function select(path: string) {
    setSelected(path);
    setView("source");
  }
  function create(event: FormEvent) {
    event.preventDefault();
    const path = name.trim();
    const parts = path.split("/");
    if (
      !path ||
      parts.some((part) => !part || part === "." || part === "..") ||
      /[\\\x00-\x1f]/.test(path) ||
      parts[0] === "SKILL.md"
    ) {
      setNameError(
        "Use a relative path such as scripts/validate.py. SKILL.md is reserved.",
      );
      return;
    }
    const occupied = [...paths, ...folders];
    if (
      occupied.includes(path) ||
      paths.some((other) => path.startsWith(other + "/")) ||
      occupied.some((other) => other.startsWith(path + "/"))
    ) {
      setNameError(
        "This path already exists or conflicts with a file or folder. Restore deleted files before reusing their paths.",
      );
      return;
    }
    if (creating === "file") {
      setFiles((previous) => ({ ...previous, [path]: "" }));
      select(path);
    } else setFolders((previous) => [...previous, path]);
    setCreating(undefined);
    setName("");
    setNameError("");
    setReview(false);
  }
  function remove() {
    if (!removing) return;
    const affected = livePaths.filter(
      (path) =>
        path === removing.path ||
        (removing.folder && path.startsWith(removing.path + "/")),
    );
    setDeletedDrafts((previous) => ({
      ...previous,
      ...Object.fromEntries(
        affected
          .filter((path) => files[path] !== undefined)
          .map((path) => [path, files[path]]),
      ),
    }));
    setFiles((previous) =>
      Object.fromEntries(
        Object.entries(previous).filter(([path]) => !affected.includes(path)),
      ),
    );
    setBinary((previous) =>
      previous.filter((path) => !affected.includes(path)),
    );
    setFolders((previous) =>
      previous.filter(
        (path) =>
          path !== removing.path && !path.startsWith(removing.path + "/"),
      ),
    );
    if (!originalPaths.includes(selected) && affected.includes(selected))
      select("SKILL.md");
    setRemoving(undefined);
  }
  function undo(path: string) {
    if (original.binary_files?.includes(path))
      setBinary((previous) => [...previous, path]);
    else
      setFiles((previous) => ({
        ...previous,
        [path]: deletedDrafts[path] ?? original.files[path],
      }));
  }
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!changed || busy) return;
    const submitter = (event.nativeEvent as SubmitEvent)
      .submitter as HTMLButtonElement | null;
    setBusy(true);
    setError("");
    try {
      const resources: Record<string, string> = Object.create(null);
      for (const path of binary) {
        const encoded = original.raw_resources?.[path];
        if (encoded === undefined)
          throw new Error(
            "Binary file data is unavailable. Reload the page before committing.",
          );
        resources[path] = encoded;
      }
      for (const [path, text] of Object.entries(files)) {
        if (path === "SKILL.md") continue;
        // Preserve exact bytes of unchanged resources, including a UTF-8 BOM.
        resources[path] =
          text === original.files[path] &&
          original.raw_resources?.[path] !== undefined
            ? original.raw_resources[path]
            : encodeText(text);
      }
      const result = await api<{ version: { version_id: string } }>(
        "commit/edited",
        "POST",
        {
          skill_id: version.skill_id,
          base_version_id: version.version_id,
          content: files["SKILL.md"],
          resources,
          message: message || null,
          version_label: mode === "manual" ? manual : null,
          version_bump: mode === "manual" ? "patch" : mode,
          active: submitter?.value === "activate",
        },
      );
      dirty(false);
      done(result.version.version_id);
    } catch (error) {
      setError((error as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const toggle = (
    <button
      type="button"
      className="sidebar-toggle"
      aria-label={showFiles ? "Hide file sidebar" : "Show file sidebar"}
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
  return (
    <form className="skill-editor" onSubmit={submit}>
      <fieldset disabled={busy}>
        <div className="row editor-heading">
          <div>
            <strong>Edit Skill</strong>
            <span className="muted">Based on {label}</span>
          </div>
          <button
            type="button"
            aria-pressed={review}
            onClick={() => setReview(!review)}
          >
            {review ? "Back to editing" : "Review changes"}
          </button>
        </div>
        <p className="editor-notice">Changes will be saved as a new version.</p>
        <div
          className={`editor-workspace ${showFiles ? "" : "files-collapsed"}`}
        >
          {showFiles && (
            <div
              className="editor-sidebar"
              role="navigation"
              aria-label="Editable Skill files"
            >
              <div className="code-title file-tree-title">
                {toggle}
                <span>Files · {livePaths.length}</span>
                <details className="editor-add">
                  <summary
                    aria-label="Add file or folder"
                    title="Add file or folder"
                  >
                    ＋
                  </summary>
                  <div>
                    {(["file", "folder"] as const).map((kind) => (
                      <button
                        key={kind}
                        type="button"
                        onClick={(event) => {
                          event.currentTarget.closest("details")!.open = false;
                          setCreating(kind);
                          setName("");
                          setNameError("");
                        }}
                      >
                        New {kind}
                      </button>
                    ))}
                  </div>
                </details>
              </div>
              <EditorTree
                paths={paths}
                folders={folders}
                selected={selected}
                changes={changes}
                select={select}
                remove={(path, folder = false) => setRemoving({ path, folder })}
                undo={undo}
              />
              {folders.some(
                (folder) =>
                  !livePaths.some((path) => path.startsWith(folder + "/")),
              ) && (
                <p className="muted empty-folder-note">
                  Empty folders are saved once they contain a file.
                </p>
              )}
            </div>
          )}
          <div className="editor-content">
            <div className="code-title">
              {!showFiles && toggle}
              <span title={selected}>{selected}</span>
              <span className="editor-file-controls">
                {review ? (
                  <span className="badge">
                    {changes.get(selected)?.change || "Unchanged"}
                  </span>
                ) : markdown && !deleted ? (
                  <>
                    <button
                      type="button"
                      className={view === "source" ? "primary" : ""}
                      onClick={() => setView("source")}
                    >
                      Source
                    </button>
                    <button
                      type="button"
                      className={view === "preview" ? "primary" : ""}
                      onClick={() => setView("preview")}
                    >
                      Preview
                    </button>
                  </>
                ) : (
                  <span className="muted">
                    {isBinary ? "Binary file" : selected.split(".").pop()}
                  </span>
                )}
              </span>
            </div>
            <div className="editor-document">
              {review ? (
                <SplitDiff
                  path={selected}
                  beforeLabel="Original"
                  afterLabel="Draft"
                  before={original.files[selected]}
                  after={files[selected]}
                  beforeNotice={isBinary ? "Binary file" : undefined}
                  afterNotice={isBinary ? "Binary file" : undefined}
                />
              ) : deleted ? (
                <Empty>
                  This file will be deleted in the new version.{" "}
                  <button type="button" onClick={() => undo(selected)}>
                    Restore file
                  </button>
                </Empty>
              ) : isBinary ? (
                <Empty>
                  Binary files are preserved unchanged. You can delete them from
                  the file sidebar.
                </Empty>
              ) : view === "preview" && markdown ? (
                <FilePreview
                  path={selected}
                  text={files[selected]}
                  paths={livePaths}
                  onOpenFile={select}
                />
              ) : (
                <div className="source-editor">
                  <div className="editor-line-numbers" aria-hidden="true">
                    {Array.from(
                      { length: files[selected]?.split("\n").length || 1 },
                      (_, index) => (
                        <div key={index}>{index + 1}</div>
                      ),
                    )}
                  </div>
                  <textarea
                    key={selected}
                    aria-label={`Edit ${selected}`}
                    spellCheck={false}
                    wrap="off"
                    value={files[selected] || ""}
                    onChange={(event) =>
                      setFiles((previous) => ({
                        ...previous,
                        [selected]: event.target.value,
                      }))
                    }
                    onScroll={(event) => {
                      const numbers =
                        event.currentTarget.previousElementSibling;
                      if (numbers)
                        numbers.scrollTop = event.currentTarget.scrollTop;
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Tab") {
                        event.preventDefault();
                        const input = event.currentTarget;
                        const start = input.selectionStart;
                        const end = input.selectionEnd;
                        const text =
                          input.value.slice(0, start) +
                          "  " +
                          input.value.slice(end);
                        setFiles((previous) => ({
                          ...previous,
                          [selected]: text,
                        }));
                        requestAnimationFrame(() =>
                          input.setSelectionRange(start + 2, start + 2),
                        );
                      }
                    }}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="editor-submit-fields">
          <label>
            Version numbering
            <select
              value={mode}
              onChange={(event) => setMode(event.target.value)}
            >
              <option value="patch">Patch +1</option>
              <option value="minor">Minor +1</option>
              <option value="major">Major +1</option>
              <option value="manual">Manual</option>
            </select>
          </label>
          {mode === "manual" && (
            <label>
              Version
              <input
                required
                pattern={versionPattern}
                placeholder="v1.2.0"
                value={manual}
                onChange={(event) => setManual(event.target.value)}
              />
            </label>
          )}
          <label>
            Commit message
            <input
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder="Describe your changes"
            />
          </label>
        </div>
        <Message text={error} error />
        <div className="editor-footer">
          <span className="muted" role="status">
            {busy
              ? "Committing…"
              : changed
                ? `● Unsaved changes · ${changes.size} ${changes.size === 1 ? "file" : "files"}`
                : "No changes"}
          </span>
          <div className="actions">
            <button type="button" onClick={cancel}>
              Cancel
            </button>
            <button
              type="submit"
              value="commit"
              disabled={!changed || !files["SKILL.md"]?.trim()}
            >
              Commit
            </button>
            <button
              type="submit"
              value="activate"
              className="primary"
              disabled={!changed || !files["SKILL.md"]?.trim()}
            >
              Commit &amp; activate
            </button>
          </div>
        </div>
      </fieldset>
      {discarding && (
        <Dialog title="Discard changes?" close={() => setDiscarding(false)}>
          <p>Your unsubmitted file changes will be discarded.</p>
          <div className="actions">
            <button type="button" onClick={() => setDiscarding(false)}>
              Keep editing
            </button>
            <button type="button" className="primary" onClick={close}>
              Discard changes
            </button>
          </div>
        </Dialog>
      )}
      {creating && (
        <Dialog title={`New ${creating}`} close={() => setCreating(undefined)}>
          <label>
            Relative path
            <input
              autoFocus
              value={name}
              placeholder={
                creating === "file" ? "scripts/validate.py" : "scripts"
              }
              onChange={(event) => {
                setName(event.target.value);
                setNameError("");
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") create(event);
              }}
            />
          </label>
          <Message text={nameError} error />
          <div className="actions">
            <button type="button" onClick={() => setCreating(undefined)}>
              Cancel
            </button>
            <button type="button" className="primary" onClick={create}>
              Create {creating}
            </button>
          </div>
        </Dialog>
      )}
      {removing && (
        <Dialog
          title={removing.folder ? "Delete folder" : "Delete file"}
          close={() => setRemoving(undefined)}
        >
          <p>
            Delete <strong>{removing.path}</strong>
            {removing.folder ? " and all files inside it" : ""} from the draft?
            Existing versions will be preserved.
          </p>
          <div className="actions">
            <button type="button" onClick={() => setRemoving(undefined)}>
              Cancel
            </button>
            <button type="button" className="primary" onClick={remove}>
              Delete
            </button>
          </div>
        </Dialog>
      )}
    </form>
  );
}
