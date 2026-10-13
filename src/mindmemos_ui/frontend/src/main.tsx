import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, session, type Connections, type Selection } from "./api";
import { useLoad, useRefresh, Message, Empty } from "./components";
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
  const connectionRefresh = useRefresh(() =>
    setConnectionRevision((value) => value + 1),
  );
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
    return () => {
      active = false;
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
            ["graphs", "Graphs"],
            ["settings", "Settings"],
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
          <div
            className="connection-status"
            aria-label="Service connection status"
          >
            <div className="connection-status-heading">
              <span>Service availability</span>
              <button type="button" onClick={connectionRefresh.refresh}>
                Refresh
              </button>
            </div>
            <Message text={connectionRefresh.error} error />
            {(["memory", "skill"] as const).map((name) => {
              const connection = connections?.[name];
              const label = connectionError
                ? "Check failed"
                : !connection
                  ? "Checking…"
                  : {
                      connected: "Available",
                      unconfigured: "Not configured",
                      unauthorized: "Authentication failed",
                      unavailable: "Unavailable",
                    }[connection.status];
              return (
                <div
                  className="connection-status-row"
                  key={name}
                  title={connection?.url}
                >
                  <span
                    className={`connection-indicator ${connection?.status || "pending"}`}
                  />
                  <span>{name === "memory" ? "Memory" : "Skill"}</span>
                  <span>{label}</span>
                </div>
              );
            })}
          </div>
          <div className="home">
            <span>SDK configuration folder</span>
            <code>{start.value?.home || "—"}</code>
          </div>
        </div>
      </aside>
      <div className="workspace">
        <header>
          <div className="page-heading">
            <span>Workspace /</span>
            {selection ? (
              <span>Skills</span>
            ) : (
              <h1>
                {page === "memories"
                  ? "Memories"
                  : page === "skills"
                    ? "Skills"
                    : page === "graphs"
                      ? "Graphs"
                      : "Settings"}
              </h1>
            )}
          </div>
          <span className="local-dot">Local</span>
        </header>
        <main className={selection ? "detail-layout" : undefined}>
          <Message text={start.error} error />
          {start.loading ? (
            <Empty>Loading…</Empty>
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
                    <Memories
                      dirty={setDirty}
                      active={!selection && page === "memories"}
                    />
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
                    dirty={setDirty}
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
                  <ServiceSettings
                    connections={connections}
                    dirty={setDirty}
                    home={start.value.home}
                    onSaved={() => setConnectionRevision((value) => value + 1)}
                  />
                ) : page === "settings" ? (
                  <ServiceSettings
                    connections={connections}
                    dirty={setDirty}
                    home={start.value.home}
                    onSaved={() => setConnectionRevision((value) => value + 1)}
                  />
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
