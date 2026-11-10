import { useState, useEffect, useRef, type FormEvent } from "react";
import { api, query } from "./api";
import { Empty, Message, useLoad, useRefresh } from "./components";

type Memory = {
  id: string;
  memory: string;
  user_id?: string | null;
  memory_type: string;
  last_update_at?: string | null;
  event_time?: string | null;
};

type MemoryPage = {
  memories: Memory[];
  page: number;
  page_size: number;
  total: number | null;
  has_more: boolean;
};

type MemorySearchResult = { memories: Memory[] };

function displayTime(value?: string | null) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "—"
    : date.toLocaleString("en-US", { hour12: false });
}

type Editing = {
  edit: (memory: Memory) => void;
  selected?: string;
  revision: number;
};

function MemoryRows({
  memories,
  showUserId = false,
  edit,
  selected,
}: { memories: Memory[]; showUserId?: boolean } & Pick<
  Editing,
  "edit" | "selected"
>) {
  return (
    <div className="table-wrap memory-table">
      <table>
        <thead>
          <tr>
            <th>Memory</th>
            {showUserId && <th>User ID</th>}
            <th>Type</th>
            <th>Updated</th>
            <th className="memory-id-heading">ID</th>
            <th>
              <span className="visually-hidden">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {memories.map((item) => (
            <tr
              key={item.id}
              className={selected === item.id ? "memory-selected" : undefined}
            >
              <td className="memory-text">
                {item.memory.length > 180 ? (
                  <details>
                    <summary>{item.memory.slice(0, 180).trimEnd()}…</summary>
                    <p>{item.memory}</p>
                  </details>
                ) : (
                  item.memory
                )}
              </td>
              {showUserId && (
                <td
                  className="memory-user-id"
                  title={item.user_id ?? undefined}
                >
                  {item.user_id || "—"}
                </td>
              )}
              <td>
                <span className="badge">{item.memory_type || "—"}</span>
              </td>
              <td className="memory-time">
                {displayTime(item.last_update_at || item.event_time)}
              </td>
              <td className="memory-id">
                <code title={item.id}>{item.id}</code>
              </td>
              <td>
                <button
                  className="memory-edit-button"
                  onClick={() => edit(item)}
                  aria-label={`Edit memory ${item.id}`}
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    aria-hidden="true"
                  >
                    <path d="m16 3 5 5-12 12-6 1 1-6Z M13 6l5 5" />
                  </svg>
                  Edit
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MemoryList({
  active,
  edit,
  selected,
  revision,
}: { active: boolean } & Editing) {
  const [page, setPage] = useState(1);
  const [reload, setReload] = useState(0);
  const refresh = useRefresh(() => setReload((current) => current + 1));
  const { value, error, loading } = useLoad<MemoryPage>(
    () =>
      active
        ? api(`memories?${query({ page: String(page), page_size: "20" })}`)
        : Promise.resolve({
            memories: [],
            page,
            page_size: 20,
            total: null,
            has_more: false,
          }),
    `${page}:${reload}:${revision}`,
    active,
  );

  return (
    <>
      <div className="memory-actions">
        <span className="muted">
          {value?.total != null
            ? `${value.total} ${value.total === 1 ? "memory" : "memories"}`
            : "Memories in the current configuration scope"}
        </span>
        <button type="button" onClick={refresh.refresh}>
          Refresh
        </button>
      </div>
      <Message text={refresh.error || error} error />
      {loading ? (
        <Empty>Loading memories…</Empty>
      ) : value?.memories.length ? (
        <>
          <MemoryRows
            memories={value.memories}
            showUserId
            edit={edit}
            selected={selected}
          />
          <div className="memory-pagination">
            <span className="muted">Page {page}</span>
            <button
              type="button"
              disabled={page === 1}
              onClick={() => setPage(page - 1)}
            >
              Previous
            </button>
            <button
              type="button"
              disabled={!value.has_more}
              onClick={() => setPage(page + 1)}
            >
              Next
            </button>
          </div>
        </>
      ) : (
        !error && <Empty>No memories</Empty>
      )}
    </>
  );
}

function MemorySearch({
  active,
  edit,
  selected,
  revision: savedRevision,
}: { active: boolean } & Editing) {
  const [input, setInput] = useState("");
  const [topK, setTopK] = useState("20");
  const [userId, setUserId] = useState("");
  const [submitted, setSubmitted] = useState<{
    query: string;
    top_k: number;
    user_id?: string;
  } | null>(null);
  const [revision, setRevision] = useState(0);
  const { value, error, loading } = useLoad<MemorySearchResult>(
    () =>
      active && submitted
        ? api("memories/search", "POST", submitted)
        : Promise.resolve({ memories: [] }),
    `${revision}:${savedRevision}`,
    active,
  );

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = input.trim();
    const limit = Number(topK);
    if (!text || !Number.isSafeInteger(limit) || limit < 1) return;
    const user = userId.trim();
    setSubmitted({
      query: text,
      top_k: limit,
      ...(user ? { user_id: user } : {}),
    });
    setRevision((current) => current + 1);
  }

  return (
    <>
      <form className="memory-search-form" onSubmit={submit}>
        <label className="memory-search-query" htmlFor="memory-query">
          Search memories
          <input
            id="memory-query"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Enter a search query"
            maxLength={2000}
            required
          />
        </label>
        <label htmlFor="memory-top-k">
          Top-K
          <input
            id="memory-top-k"
            type="number"
            min="1"
            step="1"
            required
            value={topK}
            onChange={(event) => setTopK(event.target.value)}
          />
        </label>
        <label htmlFor="memory-user-id">
          User ID filter
          <input
            id="memory-user-id"
            value={userId}
            onChange={(event) => setUserId(event.target.value)}
            placeholder="Default user"
            aria-describedby="memory-user-help"
          />
        </label>
        <button
          className="primary"
          type="submit"
          disabled={!input.trim() || loading}
        >
          {loading ? "Searching…" : "Search"}
        </button>
        <small id="memory-user-help" className="memory-search-help">
          Blank User ID uses the default user.
        </small>
      </form>
      <div className="memory-results-heading">
        Results
        {submitted && !loading && !error ? (
          <span className="muted">{value?.memories.length ?? 0} matches</span>
        ) : null}
      </div>
      <Message text={error} error />
      {!submitted ? (
        <div className="memory-search-empty">
          <svg
            width="28"
            height="28"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            aria-hidden="true"
          >
            <circle cx="10.5" cy="10.5" r="6.5" />
            <path d="m16 16 5 5" />
          </svg>
          <strong>Search your memories</strong>
          <span>Enter a query to find relevant memories.</span>
        </div>
      ) : loading ? (
        <Empty>Searching…</Empty>
      ) : value?.memories.length ? (
        <>
          <MemoryRows
            memories={value.memories}
            edit={edit}
            selected={selected}
          />
        </>
      ) : (
        !error && <Empty>No matching memories</Empty>
      )}
    </>
  );
}

export function Memories({
  active,
  dirty,
}: {
  active: boolean;
  dirty: (value: boolean) => void;
}) {
  const [tab, setTab] = useState<"list" | "search">("list");
  const [selected, setSelected] = useState<Memory>();
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState("");
  const [message, setMessage] = useState("");
  const [revision, setRevision] = useState(0);
  const opener = useRef<HTMLElement | null>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const changed = !!selected && draft !== selected.memory;
  useEffect(() => {
    dirty(active && (changed || busy));
    return () => dirty(false);
  }, [active, changed, busy, dirty]);
  useEffect(() => {
    if (!active) setSelected(undefined);
  }, [active]);
  useEffect(() => {
    if (selected) textarea.current?.focus();
  }, [selected?.id]);
  function canLeave() {
    return (
      !busy && (!changed || window.confirm("Discard unsaved memory changes?"))
    );
  }
  function close() {
    if (!canLeave()) return;
    setSelected(undefined);
    opener.current?.focus();
  }
  function edit(memory: Memory) {
    if (!canLeave()) return;
    opener.current = document.activeElement as HTMLElement;
    setSelected(memory);
    setDraft(memory.memory);
    setFailure("");
    setMessage("");
  }
  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selected || !draft.trim() || !changed || busy) return;
    setBusy(true);
    setFailure("");
    try {
      await api(`memories/${encodeURIComponent(selected.id)}`, "PUT", {
        content: draft,
      });
      setSelected(undefined);
      setRevision((value) => value + 1);
      setMessage("Memory updated.");
      dirty(false);
    } catch (cause) {
      setFailure((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const editing = { edit, selected: selected?.id, revision };
  return (
    <div className={selected ? "memories-layout editing" : "memories-layout"}>
      <div className="memories-main">
        <Message text={message} />
        <div className="tabs" role="tablist" aria-label="Memory views">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "list"}
            className={tab === "list" ? "active" : ""}
            onClick={() => {
              if (canLeave()) {
                setSelected(undefined);
                setTab("list");
              }
            }}
          >
            Browse memories
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "search"}
            className={tab === "search" ? "active" : ""}
            onClick={() => {
              if (canLeave()) {
                setSelected(undefined);
                setTab("search");
              }
            }}
          >
            Search
          </button>
        </div>
        {tab === "list" ? (
          <MemoryList active={active} {...editing} />
        ) : (
          <MemorySearch active={active} {...editing} />
        )}
      </div>
      {selected && active && (
        <aside
          className="memory-editor"
          aria-label="Edit memory"
          onKeyDown={(event) => {
            if (event.key === "Escape") close();
          }}
        >
          <div className="memory-editor-heading">
            <div>
              <h2>Edit memory</h2>
              <p className="muted">Update the selected memory.</p>
            </div>
            <button
              onClick={close}
              disabled={busy}
              aria-label="Close memory editor"
            >
              ×
            </button>
          </div>
          <form onSubmit={save}>
            <div className="memory-editor-body">
              <div className="memory-editor-metadata">
                <div>
                  <span>Memory ID</span>
                  <code>{selected.id}</code>
                </div>
                <div>
                  <span>User ID</span>
                  <p>{selected.user_id || "—"}</p>
                </div>
              </div>
              <label htmlFor="memory-content">Memory content</label>
              <textarea
                id="memory-content"
                ref={textarea}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                disabled={busy}
                required
              />
              <p className="muted">Keep the meaning clear and specific.</p>
              <div className="memory-editor-metadata">
                <div>
                  <span>Type</span>
                  <p>{selected.memory_type || "—"}</p>
                </div>
                <div>
                  <span>Last updated</span>
                  <p>
                    {displayTime(
                      selected.last_update_at || selected.event_time,
                    )}
                  </p>
                </div>
              </div>
              <Message text={failure} error />
              <p className="muted" role="status">
                {busy
                  ? "Saving…"
                  : changed
                    ? "● Unsaved changes"
                    : "No changes"}
              </p>
            </div>
            <div className="memory-editor-footer">
              <button type="button" onClick={close} disabled={busy}>
                Cancel
              </button>
              <button
                className="primary"
                disabled={busy || !changed || !draft.trim()}
              >
                {busy ? "Saving…" : "Save changes"}
              </button>
            </div>
          </form>
        </aside>
      )}
    </div>
  );
}
