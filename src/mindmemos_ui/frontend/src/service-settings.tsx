import hljs from "highlight.js/lib/core";
import json from "highlight.js/lib/languages/json";
import "highlight.js/styles/github.css";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { Dialog, Empty, Message, useLoad } from "./components";

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
}: {
  dirty: (value: boolean) => void;
  home: string;
  onSaved: () => void;
}) {
  const [showPathHelp, setShowPathHelp] = useState(false);
  const { value, error, loading } = useLoad<Document>(
    () => api<Document>(endpoint(home)),
    home,
  );

  return (
    <>
      <div className="configuration-heading">
        <h1>配置</h1>
        <span className="configuration-default-label">{home}</span>
        <button type="button" onClick={() => setShowPathHelp(true)}>
          切换配置路径
        </button>
      </div>
      {showPathHelp && (
        <Dialog title="切换配置路径" close={() => setShowPathHelp(false)}>
          <p className="muted">
            通过环境变量设置切换 MindMemOS 配置目录可以支持多个应用分身。以下命令设置本机默认目录。
          </p>
          <div className="configuration-home-commands">
            <CopyCommand
              label="执行命令"
              command={'mindmemos config home set "/Users/alice/work/.mindmemos"'}
            />
          </div>
          <p className="muted">
            执行前请将示例路径改为目标目录，执行后重新启动 MindMemOS。原配置文件不会自动移动；启动时指定的 --home 或 MINDMEMOS_HOME 优先。
          </p>
        </Dialog>
      )}
      <Message text={error} error />
      {value && !loading && !error ? (
        <ConfigurationEditor
          key={home}
          initial={value}
          home={home}
          dirty={dirty}
          onSaved={onSaved}
        />
      ) : (
        <Empty>{loading ? "加载配置…" : "无法加载此目录的配置"}</Empty>
      )}
    </>
  );
}

function CopyCommand({ label, command }: { label: string; command: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
    } catch {
      window.alert("复制失败，请手动选择命令复制。");
    }
  }

  return (
    <section>
      <h3>{label}</h3>
      <div className="configuration-command-line">
        <pre>
          <code>{command}</code>
        </pre>
        <button type="button" onClick={copy} aria-label={`复制 ${label} 命令`}>
          {copied ? "已复制" : "复制"}
        </button>
      </div>
    </section>
  );
}

function ConfigurationEditor({
  initial,
  home,
  dirty,
  onSaved,
}: {
  initial: Document;
  home: string;
  dirty: (value: boolean) => void;
  onSaved: () => void;
}) {
  const [baseline, setBaseline] = useState(initial);
  const [values, setValues] = useState(initial.values);
  const [content, setContent] = useState(initial.content);
  const [mode, setMode] = useState<"form" | "json">("form");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState("");
  const [message, setMessage] = useState("");
  const [showKeys, setShowKeys] = useState(false);
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
      !window.confirm("重新读取将放弃未保存修改，继续？")
    )
      return;
    setBusy(true);
    setFailure("");
    setMessage("");
    try {
      if (kind === "validate") {
        await normalize(mode === "json" ? content : JSON.stringify(values));
        setMessage("配置校验通过，尚未连接外部服务。");
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
          setMessage("配置已保存。");
          onSaved();
          return;
        }
        setMessage("已重新读取配置。");
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
    <section className="settings-inline" aria-label="SDK 配置">
      <div className="settings-inline-heading">
        <div>
          <h2>SDK 配置</h2>
        </div>
      </div>
      <form
        className="configuration-editor-form"
        onSubmit={(event) => {
          event.preventDefault();
          void action("save");
        }}
      >
        <fieldset disabled={busy}>
          <div className="connection-toolbar">
            <div className="actions" role="group" aria-label="配置编辑方式">
              <button
                type="button"
                aria-pressed={mode === "form"}
                onClick={() => switchMode("form")}
              >
                配置项
              </button>
              <button
                type="button"
                aria-pressed={mode === "json"}
                onClick={() => switchMode("json")}
              >
                JSON 源码
              </button>
            </div>
          </div>
          {mode === "form" ? (
            <div className="connection-fields settings-field-grid">
              <label>
                服务类型
                <select
                  value={values.service_type}
                  onChange={(event) => {
                    const type = event.target.value as Settings["service_type"];
                    setValues({
                      ...values,
                      service_type: type,
                      memory_service: {
                        ...memory,
                        base_url: type === "official" ? null : memory.base_url,
                      },
                      skill_service:
                        type === "official"
                          ? { base_url: null, auth: null }
                          : skill,
                    });
                  }}
                >
                  <option value="official">MindMemOS</option>
                  <option value="custom">本地部署</option>
                </select>
                <small>
                  本地部署需要自行配置Memory和Skill服务URL
                </small>
              </label>
              {values.service_type === "custom" && (
                <>
                  <label>
                    Memory 服务地址
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
                    <small>本地部署必填。</small>
                  </label>
                  <label>
                    Skill 服务地址（留空复用 Memory）
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
                  </label>
                </>
              )}
              <label>
                Memory API Key（可留空）
                <div className="connection-key">
                  <input
                    type={showKeys ? "text" : "password"}
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
                  <button type="button" onClick={() => setShowKeys(!showKeys)}>
                    {showKeys ? "隐藏" : "显示"}
                  </button>
                </div>
              </label>
              {values.service_type === "custom" && skill.base_url && (
                <label>
                  Skill API Key
                  <input
                    type={showKeys ? "text" : "password"}
                    required
                    autoComplete="new-password"
                    value={skill.auth?.api_key ?? ""}
                    onChange={(event) =>
                      setValues({
                        ...values,
                        skill_service: {
                          ...skill,
                          auth: { api_key: event.target.value || null },
                        },
                      })
                    }
                  />
                  <small>Skill 独立服务地址需要单独配置凭据。</small>
                </label>
              )}
              <label>
                默认用户（可留空）
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
                请求超时（秒）
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
                最大重试次数
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
          ) : (
            <>
              <p className="muted">
                完整的 SDK v2 settings.json；可编辑全部字段，保存时会校验配置。
              </p>
              <JsonEditor value={content} change={setContent} />
            </>
          )}
          <p className="muted config-path">{baseline.path}</p>
          <Message text={failure} error />
          <Message text={message} />
          <div className="configuration-editor-actions">
            <span className="muted">
              {changed
                ? "有未保存的修改"
                : baseline.exists
                  ? "已保存"
                  : "尚未创建配置文件"}
            </span>
            <div className="actions">
              <button type="button" onClick={() => action("reload")}>
                重新读取
              </button>
              <button type="button" onClick={() => action("validate")}>
                校验
              </button>
              <button className="primary">
                {busy ? "处理中…" : "保存配置"}
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
        aria-label="JSON 配置"
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
