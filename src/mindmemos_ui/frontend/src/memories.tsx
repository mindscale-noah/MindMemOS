import { useState, type FormEvent } from "react";
import { api, query } from "./api";
import { Empty, Message, useLoad } from "./components";

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
    : date.toLocaleString("zh-CN", { hour12: false });
}

function MemoryRows({ memories, showUserId = false }: { memories: Memory[]; showUserId?: boolean }) {
  return (
    <div className="table-wrap memory-table">
      <table>
        <thead>
          <tr>
            <th>记忆内容</th>
            {showUserId && <th>User ID</th>}
            <th>类型</th>
            <th>更新时间</th>
            <th>ID</th>
          </tr>
        </thead>
        <tbody>
          {memories.map((item) => (
            <tr key={item.id}>
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
              {showUserId && <td className="memory-user-id" title={item.user_id ?? undefined}>{item.user_id || "—"}</td>}
              <td><span className="badge">{item.memory_type || "—"}</span></td>
              <td className="memory-time">{displayTime(item.last_update_at || item.event_time)}</td>
              <td className="memory-id"><code title={item.id}>{item.id}</code></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MemoryList({ active }: { active: boolean }) {
  const [page, setPage] = useState(1);
  const [reload, setReload] = useState(0);
  const { value, error, loading } = useLoad<MemoryPage>(
    () =>
      active
        ? api(`memories?${query({ page: String(page), page_size: "20" })}`)
        : Promise.resolve({ memories: [], page, page_size: 20, total: null, has_more: false }),
    `${active}:${page}:${reload}`,
  );

  return (
    <>
      <div className="memory-actions">
        <span className="muted">
          {value?.total != null ? `共 ${value.total} 条记忆` : "当前配置范围的记忆"}
        </span>
        <button type="button" onClick={() => setReload((current) => current + 1)}>刷新</button>
      </div>
      <Message text={error} error />
      {loading ? (
        <Empty>加载记忆…</Empty>
      ) : value?.memories.length ? (
        <>
          <MemoryRows memories={value.memories} showUserId />
          <div className="memory-pagination">
            <span className="muted">第 {page} 页</span>
            <button type="button" disabled={page === 1} onClick={() => setPage(page - 1)}>上一页</button>
            <button type="button" disabled={!value.has_more} onClick={() => setPage(page + 1)}>下一页</button>
          </div>
        </>
      ) : (
        !error && <Empty>暂无记忆</Empty>
      )}
    </>
  );
}

function MemorySearch({ active }: { active: boolean }) {
  const [input, setInput] = useState("");
  const [topK, setTopK] = useState("20");
  const [userId, setUserId] = useState("");
  const [submitted, setSubmitted] = useState<{ query: string; top_k: number; user_id?: string } | null>(null);
  const [revision, setRevision] = useState(0);
  const { value, error, loading } = useLoad<MemorySearchResult>(
    () =>
      active && submitted
        ? api("memories/search", "POST", submitted)
        : Promise.resolve({ memories: [] }),
    `${active}:${revision}`,
  );

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = input.trim();
    const limit = Number(topK);
    if (!text || !Number.isSafeInteger(limit) || limit < 1) return;
    const user = userId.trim();
    setSubmitted({ query: text, top_k: limit, ...(user ? { user_id: user } : {}) });
    setRevision((current) => current + 1);
  }

  return (
    <>
      <form className="memory-search-form" onSubmit={submit}>
        <label htmlFor="memory-query">搜索记忆</label>
        <div className="memory-search-query">
          <input
            id="memory-query"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="输入要查找的内容"
            maxLength={2000}
            required
          />
          <button className="primary" type="submit" disabled={!input.trim()}>Search</button>
        </div>
        <div className="memory-search-filters">
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
            User ID 筛选（留空使用默认用户）
            <input
              id="memory-user-id"
              value={userId}
              onChange={(event) => setUserId(event.target.value)}
              placeholder="可选"
            />
          </label>
        </div>
      </form>
      <Message text={error} error />
      {!submitted ? (
        <Empty>输入内容后搜索当前配置范围的记忆</Empty>
      ) : loading ? (
        <Empty>搜索中…</Empty>
      ) : value?.memories.length ? (
        <>
          <p className="muted">找到 {value.memories.length} 条相关记忆</p>
          <MemoryRows memories={value.memories} />
        </>
      ) : (
        !error && <Empty>没有找到相关记忆</Empty>
      )}
    </>
  );
}

export function Memories({ active }: { active: boolean }) {
  const [tab, setTab] = useState<"list" | "search">("list");
  return (
    <>
      <h1>Memories</h1>
      <p className="muted">查看和搜索当前 Memory 服务中的记忆。</p>
      <div className="tabs" role="tablist" aria-label="Memories 功能">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "list"}
          className={tab === "list" ? "active" : ""}
          onClick={() => setTab("list")}
        >
          记忆查看
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "search"}
          className={tab === "search" ? "active" : ""}
          onClick={() => setTab("search")}
        >
          Search
        </button>
      </div>
      {tab === "list" ? <MemoryList active={active} /> : <MemorySearch active={active} />}
    </>
  );
}
