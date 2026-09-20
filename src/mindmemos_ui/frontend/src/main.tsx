import { useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { session, type Selection } from "./api";
import { useLoad, Message, Empty } from "./components";
import { Settings } from "./settings";
import { Skills } from "./skills";
import { Detail } from "./detail";
import { Graphs } from "./graphs";
import "./style.css";
function App() {
  const [page, setPage] = useState("skills");
  const [selection, setSelection] = useState<Selection>();
  const [dirty, setDirty] = useState(false);
  const start = useLoad(session, "session");
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);
  function navigate(next: string) {
    if (dirty && !window.confirm("配置尚未保存，放弃修改？")) return;
    setDirty(false);
    setSelection(undefined);
    setPage(next);
  }
  return (
    <div className="app">
      <aside>
        <div className="brand">
          <span>M</span>MindMemOS
        </div>
        <nav>
          {[
            ["skills", "Skills"],
            ["graphs", "管理图谱"],
            ["settings", "SDK 配置"],
          ].map(([key, label]) => (
            <button
              key={key}
              className={page === key ? "active" : ""}
              onClick={() => navigate(key)}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="home">
          <span>本地工作区</span>
          <code>{start.value?.home || "—"}</code>
        </div>
      </aside>
      <div className="workspace">
        <header>
          <span>
            工作区 /{" "}
            {page === "skills"
              ? "Skills"
              : page === "graphs"
                ? "管理图谱"
                : "配置"}
          </span>
          <span className="local-dot">本机</span>
        </header>
        <main className={selection ? "detail-layout" : undefined}>
          <Message text={start.error} error />
          {start.loading ? (
            <Empty>加载中…</Empty>
          ) : (
            start.value &&
            (selection ? (
              <Detail
                key={selection.source + selection.id + selection.version}
                selection={selection}
                back={() => setSelection(undefined)}
              />
            ) : page === "settings" ? (
              <Settings dirty={setDirty} />
            ) : page === "graphs" ? (
              <Graphs
                connected={start.value.graphs_connected}
                open={setSelection}
              />
            ) : (
              <Skills open={setSelection} />
            ))
          )}
        </main>
      </div>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
