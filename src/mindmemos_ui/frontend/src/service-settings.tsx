import { useEffect, useMemo, useRef, useState } from "react";
import hljs from "highlight.js/lib/core";
import yaml from "highlight.js/lib/languages/yaml";
import "highlight.js/styles/github.css";
import { api } from "./api";
import { Dialog, Empty, Message, useLoad } from "./components";

hljs.registerLanguage("yaml", yaml);
type Kind = "memory" | "skill";
type Entry = { name: string; path: string; exists: boolean };
type Connection = {
  base_url: string;
  api_key: string | null;
  timeout_seconds: number;
  user_id?: string | null;
  mode?: "local" | "remote";
  experiment_dir?: string;
};
type Document = Entry & {
  kind: Kind;
  content: string;
  values: Connection;
  revision: string | null;
  draft?: { content: string; values: Connection };
};
type Library = {
  home: string;
  runtime_home: string;
  memory: Entry[];
  skill: Entry[];
};
const labels = { memory: "Memory", skill: "Skill" };
function endpoint(kind: Kind, name: string, home: string, suffix = "") {
  return `client-configurations/${kind}/${encodeURIComponent(name)}${suffix}?${new URLSearchParams({ home })}`;
}

export function ServiceSettings({
  dirty,
}: {
  dirty: (value: boolean) => void;
}) {
  const [home, setHome] = useState("");
  const [homeInput, setHomeInput] = useState(home);
  const [pickingDirectory, setPickingDirectory] = useState(false);
  const [manualHome, setManualHome] = useState(false);
  const [revision, setRevision] = useState(0);
  const { value, error, loading } = useLoad<Library | null>(async () => {
    const selected =
      home ||
      (await api<{ home: string | null }>("configurations/default-home")).home;
    return selected
      ? api<Library>(
          `client-configurations?${new URLSearchParams({ home: selected })}`,
        )
      : null;
  }, `${home}:${revision}`);
  const [document, setDocument] = useState<Document>();
  const [editorHome, setEditorHome] = useState("");
  const [failure, setFailure] = useState("");
  const [opening, setOpening] = useState(false);
  const [exporting, setExporting] = useState<Kind>();
  const importInput = useRef<HTMLInputElement>(null);
  const importKind = useRef<Kind>("skill");
  useEffect(() => {
    if (value) setHomeInput(value.home);
  }, [value?.home]);

  function selectHome(next: string) {
    setHomeInput(next);
    setHome(next);
    setRevision((v) => v + 1);
  }

  async function chooseHome() {
    setPickingDirectory(true);
    setFailure("");
    try {
      const result = await api<{ path: string | null; unavailable: boolean }>(
        "configurations/choose-directory",
        "POST",
        { home: value?.home || home || null },
      );
      if (result.unavailable) {
        setManualHome(true);
        setFailure(
          "系统目录选择器不可用或已有窗口打开。请检查系统窗口，或手动输入目录路径。",
        );
      } else if (result.path) {
        selectHome(result.path);
        setManualHome(false);
      }
    } catch {
      setManualHome(true);
      setFailure("无法打开系统目录选择器，请手动输入目录路径。");
    } finally {
      setPickingDirectory(false);
    }
  }

  async function open(
    kind: Kind,
    name: string,
    draft?: { content: string; values: Connection },
  ) {
    if (!value || loading) return;
    setOpening(true);
    setFailure("");
    try {
      const next = await api<Document>(endpoint(kind, name, value.home));
      setEditorHome(value.home);
      setDocument({ ...next, draft });
    } catch (e) {
      setFailure((e as Error).message);
    } finally {
      setOpening(false);
    }
  }
  async function importFile(file: File) {
    if (file.size > 2000000) {
      setFailure("配置文件不能超过 2 MB。");
      return;
    }
    try {
      const data = await api<{ content: string; values: Connection }>(
        `client-configurations/${importKind.current}/normalize`,
        "POST",
        { content: await file.text() },
      );
      await open(importKind.current, "default", data);
    } catch (e) {
      setFailure((e as Error).message);
    }
  }
  return (
    <>
      <h1>配置</h1>
      <p className="muted">管理 Memory 与 Skill 的服务连接。</p>
      <form
        className="configuration-home"
        onSubmit={(event) => {
          event.preventDefault();
          if (manualHome && !pickingDirectory) selectHome(homeInput.trim());
        }}
      >
        <label htmlFor="configuration-home">HOME 路径</label>
        <input
          id="configuration-home"
          value={homeInput}
          readOnly={!manualHome}
          disabled={pickingDirectory}
          onChange={(event) => setHomeInput(event.target.value)}
          placeholder="加载默认目录…"
          required
        />
        <button
          type="button"
          onClick={chooseHome}
          disabled={loading || pickingDirectory}
        >
          {pickingDirectory ? "选择中…" : "选择目录…"}
        </button>
        {manualHome && (
          <button disabled={loading || pickingDirectory}>使用此路径</button>
        )}
      </form>
      <p className="muted">
        {pickingDirectory
          ? "请在系统弹出的窗口中选择目录；取消后保持原目录。"
          : "选择本机配置目录；当前运行工作区保持不变。"}{" "}
        <button
          type="button"
          className="manual-home-toggle"
          disabled={pickingDirectory}
          onClick={() => {
            setManualHome(!manualHome);
            setHomeInput(value?.home || home);
          }}
        >
          {manualHome ? "收起手动输入" : "手动输入路径"}
        </button>
      </p>
      <Message text={error || failure} error />
      <div className="configuration-columns">
        {(["memory", "skill"] as Kind[]).map((kind) => {
          const available = Boolean(value && !loading && !error);
          const entry = available
            ? value?.[kind].find((item) => item.name === "default")
            : undefined;
          return (
            <section
              className={`configuration-card${available ? "" : " configuration-card-disabled"}`}
              aria-disabled={!available}
              key={kind}
            >
              <div className="configuration-card-heading">
                <h2>{labels[kind]} 配置</h2>
                <div className="actions">
                  <button
                    disabled={!available || opening}
                    onClick={() => {
                      importKind.current = kind;
                      importInput.current?.click();
                    }}
                  >
                    导入配置
                  </button>
                  <a
                    className="configuration-template"
                    href={`/api/client-configurations/${kind}/template`}
                    download
                  >
                    完整 YAML 模板
                  </a>
                  <button
                    disabled={!available || !entry?.exists || opening}
                    onClick={() => setExporting(kind)}
                  >
                    导出配置
                  </button>
                </div>
              </div>
              <p className="muted">
                {kind === "memory"
                  ? "SDK 服务地址、凭据与连接选项"
                  : "服务连接与实验输出目录"}
              </p>
              {!available ? (
                <Empty>{loading ? "加载配置…" : "无法加载此目录的配置"}</Empty>
              ) : entry?.exists ? (
                <button
                  className="configuration-entry"
                  disabled={opening}
                  onClick={() => open(kind, "default")}
                >
                  <span className={`config-file-icon ${kind}`}>
                    {kind === "memory" ? "M" : "S"}
                  </span>
                  <span className="configuration-entry-info">
                    <strong>{labels[kind]} 配置</strong>
                    <span>
                      {kind === "memory"
                        ? "settings.json"
                        : "skill-client.yaml"}{" "}
                      · 点击编辑
                    </span>
                  </span>
                  <span aria-hidden="true">›</span>
                </button>
              ) : (
                <>
                  <Empty>此目录尚无 {labels[kind]} 配置</Empty>
                  <button
                    className="configuration-create"
                    disabled={opening}
                    onClick={() => open(kind, "default")}
                  >
                    ＋ 创建配置
                  </button>
                </>
              )}
            </section>
          );
        })}
      </div>
      <input
        ref={importInput}
        type="file"
        accept=".yaml,.yml"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file) void importFile(file);
        }}
      />
      {exporting && value && (
        <Dialog
          title={`导出 ${labels[exporting]} 配置`}
          close={() => setExporting(undefined)}
        >
          <p className="muted">
            下载 YAML，包含已保存的连接设置与凭据，请妥善保管。
          </p>
          {value[exporting]
            .filter((entry) => entry.name === "default" && entry.exists)
            .map((entry) => (
              <p key={entry.name}>
                <a
                  href={`/api/${endpoint(exporting, entry.name, value.home, "/export")}`}
                  download
                >
                  {labels[exporting]} 配置 · 下载配置
                </a>
              </p>
            ))}
          {!value[exporting].some((entry) => entry.exists) && (
            <Empty>尚无已保存的配置</Empty>
          )}
        </Dialog>
      )}
      {document && (
        <ConfigurationEditor
          initial={document}
          home={editorHome}
          dirty={dirty}
          close={() => {
            setDocument(undefined);
            dirty(false);
          }}
          saved={() => setRevision((v) => v + 1)}
        />
      )}
    </>
  );
}

function ConfigurationEditor({
  initial,
  home,
  dirty,
  close,
  saved,
}: {
  initial: Document;
  home: string;
  dirty: (value: boolean) => void;
  close: () => void;
  saved: () => void;
}) {
  const [baseline, setBaseline] = useState(initial);
  const [values, setValues] = useState(initial.draft?.values ?? initial.values);
  const [content, setContent] = useState(
    initial.draft?.content ?? initial.content,
  );
  const [mode, setMode] = useState<"form" | "yaml">("form");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState("");
  const [message, setMessage] = useState("");
  const [showKey, setShowKey] = useState(false);
  const importFile = useRef<HTMLInputElement>(null);
  const changed =
    !baseline.exists ||
    (mode === "yaml"
      ? content !== baseline.content
      : JSON.stringify(values) !== JSON.stringify(baseline.values));
  useEffect(() => {
    dirty(changed);
  }, [changed]);
  function dismiss() {
    if (!busy && (!changed || window.confirm("配置尚未保存，放弃修改？")))
      close();
  }
  function update(patch: Partial<Connection>) {
    setValues({ ...values, ...patch });
    setMessage("");
  }
  async function normalize(text: string) {
    return api<{ content: string; values: Connection }>(
      `client-configurations/${initial.kind}/normalize`,
      "POST",
      { content: text },
    );
  }
  async function switchMode(next: "form" | "yaml") {
    if (next === mode) return;
    setBusy(true);
    setFailure("");
    try {
      const result = await normalize(
        mode === "yaml" ? content : JSON.stringify(values),
      );
      setValues(result.values);
      setContent(result.content);
      setMode(next);
    } catch (e) {
      setFailure((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function importYaml(file: File) {
    setBusy(true);
    setFailure("");
    try {
      if (file.size > 1000000) throw new Error("配置文件不能超过 1 MB。");
      const result = await normalize(await file.text());
      setValues(result.values);
      setContent(result.content);
      setMode("form");
      setMessage("已导入到表单，请检查后保存。尚未写入配置文件。");
    } catch (e) {
      setFailure((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function action(action: "save" | "validate" | "reload") {
    if (
      action === "reload" &&
      changed &&
      !window.confirm("重新读取将放弃未保存修改，继续？")
    )
      return;
    setBusy(true);
    setFailure("");
    setMessage("");
    try {
      if (action === "validate") {
        await normalize(mode === "yaml" ? content : JSON.stringify(values));
        setMessage("配置校验通过，尚未连接外部服务。");
      } else {
        const next = await api<Document>(
          endpoint(initial.kind, "default", home),
          action === "save" ? "PUT" : "GET",
          action === "save"
            ? {
                content: mode === "yaml" ? content : JSON.stringify(values),
                revision: baseline.revision,
              }
            : undefined,
        );
        setBaseline(next);
        setValues(next.values);
        setContent(next.content);
        dirty(false);
        saved();
        if (action === "save") {
          close();
          return;
        }
        setMessage("已重新读取配置。");
      }
    } catch (e) {
      setFailure((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog title={`${labels[initial.kind]} 配置`} close={dismiss}>
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
                逐项填写
              </button>
              <button
                type="button"
                aria-pressed={mode === "yaml"}
                onClick={() => switchMode("yaml")}
              >
                YAML 编辑
              </button>
            </div>
            <div className="actions">
              <button type="button" onClick={() => importFile.current?.click()}>
                导入 YAML
              </button>
              <a
                href={`/api/client-configurations/${initial.kind}/template`}
                download
              >
                下载完整模板
              </a>
            </div>
          </div>
          <input
            ref={importFile}
            type="file"
            hidden
            accept=".yaml,.yml"
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) void importYaml(file);
            }}
          />
          {mode === "form" ? (
            <div className="connection-fields">
              {initial.kind === "skill" && (
                <label>
                  连接模式
                  <select
                    value={values.mode}
                    onChange={(event) =>
                      update({ mode: event.target.value as "local" | "remote" })
                    }
                  >
                    <option value="local">本地服务</option>
                    <option value="remote">远程服务</option>
                  </select>
                  <small>
                    本地模式连接本机 FastAPI；项目由服务启动时自动分配。
                  </small>
                </label>
              )}
              <label>
                服务地址
                <input
                  type="url"
                  required
                  value={values.base_url}
                  onChange={(event) => update({ base_url: event.target.value })}
                />
                <small>
                  {initial.kind === "skill"
                    ? "填写 Skill FastAPI 地址。本地模式需使用 localhost、127.0.0.1 或 ::1。"
                    : "填写 Memory FastAPI 地址。"}
                </small>
              </label>
              <label>
                API Key（可留空）
                <div className="connection-key">
                  <input
                    type={showKey ? "text" : "password"}
                    autoComplete="new-password"
                    value={values.api_key ?? ""}
                    onChange={(event) =>
                      update({ api_key: event.target.value || null })
                    }
                  />
                  <button type="button" onClick={() => setShowKey(!showKey)}>
                    {showKey ? "隐藏" : "显示"}
                  </button>
                </div>
                <small>
                  {initial.kind === "skill"
                    ? "本地免鉴权服务可留空；远程服务按需填写。项目归属由服务端确定，无需填写 project_id。"
                    : "填写服务访问凭据；清空后保存会删除已存凭据。"}
                </small>
              </label>
              {initial.kind === "memory" && (
                <label>
                  默认用户（可留空）
                  <input
                    value={values.user_id ?? ""}
                    onChange={(event) =>
                      update({ user_id: event.target.value || null })
                    }
                  />
                  <small>仅在 Memory 接口需要指定默认用户时填写。</small>
                </label>
              )}
              <label>
                请求超时（秒）
                <input
                  type="number"
                  min={1}
                  max={604800}
                  required
                  value={values.timeout_seconds}
                  onChange={(event) =>
                    update({ timeout_seconds: Number(event.target.value) })
                  }
                />
                <small>
                  默认 21600 秒（6 小时）。请求失败后不会自动重试算法任务。
                </small>
              </label>
              {initial.kind === "skill" && (
                <label>
                  实验目录
                  <input
                    required
                    value={values.experiment_dir ?? ""}
                    onChange={(event) =>
                      update({ experiment_dir: event.target.value })
                    }
                  />
                  <small>
                    记录实验输出与日志。相对路径基于当前 HOME（{home}
                    ）；不是远端服务器的文件路径。
                  </small>
                </label>
              )}
            </div>
          ) : (
            <>
              <p className="muted">
                下面包含当前全部连接字段。可直接编辑或下载默认完整模板；导入后可切回逐项填写检查。不填写
                project_id 或 max_retries。
              </p>
              <YamlEditor value={content} change={setContent} />
            </>
          )}
          <p className="muted config-path">{baseline.path}</p>
          <Message text={failure} error />
          <Message text={message} />
          <div className="configuration-editor-actions">
            <span className="muted">
              {changed ? "有未保存的修改" : "已保存"}
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
    </Dialog>
  );
}

function YamlEditor({
  value,
  change,
}: {
  value: string;
  change: (value: string) => void;
}) {
  const preview = useRef<HTMLPreElement>(null);
  const html = useMemo(
    () => hljs.highlight(value + "\n", { language: "yaml" }).value,
    [value],
  );
  return (
    <div className="yaml-editor">
      <pre ref={preview} aria-hidden="true">
        <code className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
      </pre>
      <textarea
        aria-label="YAML 配置"
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
