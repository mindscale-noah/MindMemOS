import { useState } from "react";
import { createRoot } from "react-dom/client";
import { session, type Selection } from "./api";
import { useLoad, Message, Empty } from "./components";
import { ServiceSettings } from "./service-settings";
import { Skills } from "./skills";
import { Detail } from "./detail";
import { Graphs } from "./graphs";
import { useNavigation, type Page } from "./navigation";
import "./style.css";
function App() {
  const { route, navigate, back, setDirty } = useNavigation();
  const { page, selection } = route;
  const start = useLoad(session, "session");
  const [visited, setVisited] = useState(() => new Set([page]));
  if (!visited.has(page)) setVisited(new Set([...visited, page]));
  function open(selection: Selection) {
    navigate({ page, selection });
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
            ["settings", "配置"],
          ].map(([key, label]) => (
            <button
              key={key}
              className={page === key ? "active" : ""}
              onClick={() => navigate({ page: key as Page })}
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
            start.value && (
              <>
                {visited.has("skills") && (
                  <div hidden={!!selection || page !== "skills"}>
                    <Skills
                      open={open}
                      active={!selection && page === "skills"}
                    />
                  </div>
                )}
                {visited.has("graphs") && (
                  <div hidden={!!selection || page !== "graphs"}>
                    <Graphs
                      connected={start.value.graphs_connected}
                      open={open}
                    />
                  </div>
                )}
                {selection ? (
                  <Detail
                    key={selection.source + selection.id}
                    selection={selection}
                    tab={
                      route.tab || (selection.version ? "content" : "versions")
                    }
                    change={(tab, version, replace) =>
                      navigate(
                        {
                          page,
                          selection: { ...selection, version },
                          tab,
                        },
                        replace,
                      )
                    }
                    back={back}
                  />
                ) : page === "service-settings" ? (
                  <ServiceSettings dirty={setDirty} />
                ) : page === "settings" ? (
                  <ServiceSettings dirty={setDirty} />
                ) : null}
              </>
            )
          )}
        </main>
      </div>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
