import hljs from "highlight.js/lib/core";
import json from "highlight.js/lib/languages/json";
import "highlight.js/styles/github.css";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Connections } from "./api";
import { Empty, Message, useLoad } from "./components";

hljs.registerLanguage("json", json);

type Settings = {
  version: number;
  service_type: "official" | "custom";
  memory_service: {
    base_url?: string | null;
    auth: { api_key?: string | null };
  };
  skill_service: {
    base_url?: string | null;
    auth?: { api_key?: string | null } | null;
  };
  defaults: { user_id?: string | null };
  network: { timeout_seconds: number; max_retries: number };
  metadata: { created_at?: string | null; updated_at?: string | null };
};
type Document = {
  path: string;
  exists: boolean;
  revision: string | null;
  content: string;
  values: Settings;
};
type Draft = Pick<Document, "content" | "values">;

function endpoint(home: string, suffix = "") {
  return `client-configurations/sdk${suffix}?${new URLSearchParams({ home })}`;
}

export function ServiceSettings({
  dirty,
  home,
  onSaved,
  connections,
}: {
  dirty: (value: boolean) => void;
  home: string;
  onSaved: () => void;
  connections?: Connections | null;
}) {
  const { value, error, loading } = useLoad<Document>(
    () => api<Document>(endpoint(home)),
    home,
  );

  return (
    <>
      <div className="configuration-heading">
        <span className="configuration-default-label">{home}</span>
      </div>
      <Message text={error} error />
      {value && !loading && !error ? (
        <ConfigurationEditor
          key={home}
          initial={value}
          home={home}
          dirty={dirty}
          onSaved={onSaved}
          connections={connections}
        />
      ) : (
        <Empty>
          {loading
            ? "Loading configuration…"
            : "Unable to load configuration from this folder"}
        </Empty>
      )}
    </>
  );
}

function ConfigurationEditor({
  initial,
  home,
  dirty,
  onSaved,
  connections,
}: {
  initial: Document;
  home: string;
  dirty: (value: boolean) => void;
  onSaved: () => void;
  connections?: Connections | null;
}) {
  const [baseline, setBaseline] = useState(initial);
  const [values, setValues] = useState(initial.values);
  const [content, setContent] = useState(initial.content);
  const [mode, setMode] = useState<"form" | "json">("form");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState("");
  const [message, setMessage] = useState("");
  const [showMemoryKey, setShowMemoryKey] = useState(false);
  const [showSkillKey, setShowSkillKey] = useState(false);
  const changed =
    mode === "json"
      ? content !== baseline.content
      : JSON.stringify(values) !== JSON.stringify(baseline.values);

  useEffect(() => {
    dirty(changed);
  }, [changed, dirty]);

  async function normalize(source: string) {
    return api<Draft>("client-configurations/sdk/normalize", "POST", {
      content: source,
    });
  }

  async function switchMode(next: "form" | "json") {
    if (next === mode) return;
    setBusy(true);
    setFailure("");
    try {
      const result = await normalize(
        mode === "json" ? content : JSON.stringify(values),
      );
      setValues(result.values);
      setContent(result.content);
      setMode(next);
    } catch (cause) {
      setFailure((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function action(kind: "save" | "validate" | "reload") {
    if (
      kind === "reload" &&
      changed &&
      !window.confirm("Reload and discard unsaved changes?")
    )
      return;
    setBusy(true);
    setFailure("");
    setMessage("");
    try {
      if (kind === "validate") {
        await normalize(mode === "json" ? content : JSON.stringify(values));
        setMessage(
          "Configuration is valid. External service connections have not been tested.",
        );
      } else {
        const next = await api<Document>(
          endpoint(home),
          kind === "save" ? "PUT" : "GET",
          kind === "save"
            ? {
                content: mode === "json" ? content : JSON.stringify(values),
                revision: baseline.revision,
              }
            : undefined,
        );
        setBaseline(next);
        setValues(next.values);
        setContent(next.content);
        dirty(false);
        if (kind === "save") {
          setMessage("Configuration saved.");
          onSaved();
          return;
        }
        setMessage("Configuration reloaded.");
      }
    } catch (cause) {
      setFailure((cause as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const memory = values.memory_service;
  const skill = values.skill_service;
  return (
    <section className="settings-inline" aria-label="SDK configuration">
      <form
        className="configuration-editor-form"
        onSubmit={(event) => {
          event.preventDefault();
          void action("save");
        }}
      >
        <fieldset disabled={busy}>
          <div className="connection-toolbar">
            <h2>SDK configuration</h2>
            <div
              className="actions"
              role="group"
              aria-label="Configuration editor mode"
            >
              <button
                type="button"
                aria-pressed={mode === "form"}
                onClick={() => switchMode("form")}
              >
                Fields
              </button>
              <button
                type="button"
                aria-pressed={mode === "json"}
                onClick={() => switchMode("json")}
              >
                JSON source
              </button>
            </div>
          </div>
          {mode === "form" ? (
            <div className="settings-sections">
              <section
                className="settings-card settings-service-type"
                aria-label="Service type"
              >
                {" "}
                <label>
                  Service type
                  <select
                    value={values.service_type}
                    onChange={(event) => {
                      const type = event.target
                        .value as Settings["service_type"];
                      setValues({
                        ...values,
                        service_type: type,
                        memory_service: {
                          ...memory,
                          base_url:
                            type === "official" ? null : memory.base_url,
                        },
                        skill_service:
                          type === "official"
                            ? { base_url: null, auth: null }
                            : skill,
                      });
                    }}
                  >
                    <option value="official">MindMemOS</option>
                    <option value="custom">Self-hosted</option>
                  </select>
                  <small>
                    Set the service URLs and credentials below.
                  </small>
                </label>
              </section>
              <div className="settings-services">
                <section className="settings-card" aria-label="Memory service">
                  <div className="settings-service-heading">
                    <h3>Memory service</h3>
                    <ServiceConnection
                      connection={connections?.memory}
                      changed={changed}
                    />
                  </div>
                  {values.service_type === "custom" && (
                    <label>
                      Service URL
                      <input
                        type="url"
                        required
                        value={memory.base_url ?? ""}
                        onChange={(event) =>
                          setValues({
                            ...values,
                            memory_service: {
                              ...memory,
                              base_url: event.target.value || null,
                            },
                          })
                        }
                      />
                      <small>Required for self-hosted deployment.</small>
                    </label>
                  )}
                  <label>
                    API key (optional)
                    <div className="connection-key">
                      <input
                        type={showMemoryKey ? "text" : "password"}
                        autoComplete="new-password"
                        value={memory.auth.api_key ?? ""}
                        onChange={(event) =>
                          setValues({
                            ...values,
                            memory_service: {
                              ...memory,
                              auth: { api_key: event.target.value || null },
                            },
                          })
                        }
                      />
                      <button
                        type="button"
                        aria-label={
                          showMemoryKey
                            ? "Hide Memory API key"
                            : "Show Memory API key"
                        }
                        onClick={() => setShowMemoryKey(!showMemoryKey)}
                      >
                        <KeyVisibilityIcon visible={showMemoryKey} />
                        {showMemoryKey ? "Hide" : "Show"}
                      </button>
                    </div>
                  </label>
                </section>
                <section className="settings-card" aria-label="Skill service">
                  <div className="settings-service-heading">
                    <h3>Skill service</h3>
                    <ServiceConnection
                      connection={connections?.skill}
                      changed={changed}
                    />
                  </div>
                  {values.service_type === "custom" ? (
                    <>
                      <label>
                        Service URL
                        <input
                          type="url"
                          value={skill.base_url ?? ""}
                          onChange={(event) =>
                            setValues({
                              ...values,
                              skill_service: event.target.value
                                ? {
                                    base_url: event.target.value,
                                    auth: skill.auth ?? { api_key: null },
                                  }
                                : { base_url: null, auth: null },
                            })
                          }
                        />
                        <small>Leave blank to use the Memory service.</small>
                      </label>

                      {skill.base_url ? (
                        <label>
                          API key
                          <div className="connection-key">
                            <input
                              type={showSkillKey ? "text" : "password"}
                              required
                              autoComplete="new-password"
                              value={skill.auth?.api_key ?? ""}
                              onChange={(event) =>
                                setValues({
                                  ...values,
                                  skill_service: {
                                    ...skill,
                                    auth: {
                                      api_key: event.target.value || null,
                                    },
                                  },
                                })
                              }
                            />
                            <button
                              type="button"
                              aria-label={
                                showSkillKey
                                  ? "Hide Skill API key"
                                  : "Show Skill API key"
                              }
                              onClick={() => setShowSkillKey(!showSkillKey)}
                            >
                              <KeyVisibilityIcon visible={showSkillKey} />
                              {showSkillKey ? "Hide" : "Show"}
                            </button>
                          </div>
                          <small>
                            A separate Skill service URL requires its own
                            credentials.
                          </small>
                        </label>
                      ) : (
                        <p className="muted settings-inherited">
                          Uses the Memory service URL and API key.
                        </p>
                      )}
                    </>
                  ) : (
                    <p className="muted settings-inherited">
                      Uses the MindMemOS service and Memory API key.
                    </p>
                  )}
                </section>
              </div>
              <section
                className="settings-card"
                aria-label="Defaults and network"
              >
                <h3>Defaults &amp; network</h3>
                <div className="settings-defaults">
                  {" "}
                  <label>
                    Default user (optional)
                    <input
                      value={values.defaults.user_id ?? ""}
                      onChange={(event) =>
                        setValues({
                          ...values,
                          defaults: { user_id: event.target.value || null },
                        })
                      }
                    />
                  </label>
                  <label>
                    Request timeout (seconds)
                    <input
                      type="number"
                      min={1}
                      required
                      value={values.network.timeout_seconds}
                      onChange={(event) =>
                        setValues({
                          ...values,
                          network: {
                            ...values.network,
                            timeout_seconds: Number(event.target.value),
                          },
                        })
                      }
                    />
                  </label>
                  <label>
                    Maximum retries
                    <input
                      type="number"
                      min={0}
                      required
                      value={values.network.max_retries}
                      onChange={(event) =>
                        setValues({
                          ...values,
                          network: {
                            ...values.network,
                            max_retries: Number(event.target.value),
                          },
                        })
                      }
                    />
                  </label>
                </div>
              </section>
            </div>
          ) : (
            <>
              <p className="muted">
                Complete SDK v2 settings.json. All fields are editable and
                validated when saved.
              </p>
              <JsonEditor value={content} change={setContent} />
            </>
          )}
          <p className="muted config-path">{baseline.path}</p>
          <Message text={failure} error />
          <Message text={message} />
          <div className="configuration-editor-actions">
            <span className={changed || !baseline.exists ? "muted" : "settings-saved"}>
              {!changed && baseline.exists && <span aria-hidden="true">✓ </span>}
              {changed
                ? "Unsaved changes"
                : baseline.exists
                  ? "Saved"
                  : "Configuration file has not been created"}
            </span>
            <div className="actions">
              <button type="button" onClick={() => action("reload")}>
                Reload
              </button>
              <button type="button" onClick={() => action("validate")}>
                Validate
              </button>
              <button className="primary">
                {busy ? "Processing…" : "Save configuration"}
              </button>
            </div>
          </div>
        </fieldset>
      </form>
    </section>
  );
}

function JsonEditor({
  value,
  change,
}: {
  value: string;
  change: (value: string) => void;
}) {
  const preview = useRef<HTMLPreElement>(null);
  const html = useMemo(
    () => hljs.highlight(value + "\n", { language: "json" }).value,
    [value],
  );
  return (
    <div className="yaml-editor">
      <pre ref={preview} aria-hidden="true">
        <code className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
      <textarea
        aria-label="JSON configuration"
        value={value}
        required
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        wrap="off"
        onChange={(event) => change(event.target.value)}
        onScroll={(event) => {
          if (preview.current) {
            preview.current.scrollTop = event.currentTarget.scrollTop;
            preview.current.scrollLeft = event.currentTarget.scrollLeft;
          }
        }}
      />
    </div>
  );
}

function ServiceConnection({
  connection,
  changed,
}: {
  connection?: Connections["memory"];
  changed: boolean;
}) {
  const status = changed ? "pending" : connection?.status || "pending";
  const label = changed
    ? "Unsaved changes"
    : connection
      ? {
          connected: "Connected",
          unconfigured: "Not configured",
          unauthorized: "Authentication failed",
          unavailable: "Unavailable",
        }[connection.status]
      : "Checking…";
  return (
    <span className="settings-service-status">
      <span className={`connection-indicator ${status}`} />
      {label}
    </span>
  );
}

function KeyVisibilityIcon({ visible }: { visible: boolean }) {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" /><circle cx="12" cy="12" r="3" />{visible && <path d="m3 3 18 18" />}</svg>;
}
