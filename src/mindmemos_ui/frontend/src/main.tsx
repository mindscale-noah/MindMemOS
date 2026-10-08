import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, session, type Connections, type Selection } from "./api";
import { useLoad, Message, Empty } from "./components";
import { ServiceSettings } from "./service-settings";
import { Skills } from "./skills";
import { Memories } from "./memories";
import { Detail } from "./detail";
import { Graphs } from "./graphs";
import { useNavigation, type Page } from "./navigation";
import "./style.css";
function App() {
  const { route, navigate, back, setDirty } = useNavigation();
  const { page, selection } = route;
  const start = useLoad(session, "session");
  const [connections, setConnections] = useState<Connections | null>(null);
  const [connectionError, setConnectionError] = useState(false);
  const [connectionRevision, setConnectionRevision] = useState(0);
  useEffect(() => {
    if (!start.value) return;
    let active = true;
    async function refresh() {
      try {
        const result = await api<Connections>("connections");
        if (active) {
          setConnections(result);
          setConnectionError(false);
        }
      } catch {
        if (active) {
          setConnections(null);
          setConnectionError(true);
        }
      }
    }
    void refresh();
    const interval = window.setInterval(refresh, 30000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [start.value, connectionRevision]);
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
            ["memories", "Memories"],
            ["skills", "Skills"],
            ["graphs", "SkillSets"],
            ["settings", "Configs"],
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
        <div className="sidebar-footer">
          <div className="connection-status" aria-label="服务连接状态">
            <div className="connection-status-heading">
              <span>服务连接</span>
              <button type="button" onClick={() => setConnectionRevision((value) => value + 1)}>
                刷新
              </button>
            </div>
            {(["memory", "skill"] as const).map((name) => {
              const connection = connections?.[name];
              const label = connectionError
                ? "检查失败"
                : !connection
                  ? "检查中…"
                  : {
                      connected: "已连接",
                      unconfigured: "未配置",
                      unauthorized: "认证失败",
                      unavailable: "无法连接",
                    }[connection.status];
              return (
                <div className="connection-status-row" key={name} title={connection?.url}>
                  <span className={`connection-indicator ${connection?.status || "pending"}`} />
                  <span>{name === "memory" ? "Memory" : "Skill"}</span>
                  <span>{label}</span>
                </div>
              );
            })}
          </div>
          <div className="home">
            <span>SDK 配置目录</span>
            <code>{start.value?.home || "—"}</code>
          </div>
        </div>
      </aside>
      <div className="workspace">
        <header>
          <span>
            工作区 /{" "}
            {page === "memories"
              ? "Memories"
              : page === "skills"
              ? "Skills"
              : page === "graphs"
                ? "SkillSets"
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
                {visited.has("memories") && (
                  <div hidden={!!selection || page !== "memories"}>
                    <Memories active={!selection && page === "memories"} />
                  </div>
                )}
                {visited.has("graphs") && (
                  <div hidden={!!selection || page !== "graphs"}>
                    <Graphs
                      connected={connections?.skill.status === "connected"}
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
                  <ServiceSettings dirty={setDirty} home={start.value.home} onSaved={() => setConnectionRevision((value) => value + 1)} />
                ) : page === "settings" ? (
                  <ServiceSettings dirty={setDirty} home={start.value.home} onSaved={() => setConnectionRevision((value) => value + 1)} />
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
