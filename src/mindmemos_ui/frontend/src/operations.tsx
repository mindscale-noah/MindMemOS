import React, { useState } from "react";
import { api, versionPattern, type Ref } from "./api";
import { useLoad, Message, Dialog, Changes } from "./components";
export function Register({
  close,
  done,
}: {
  close: () => void;
  done: () => void;
}) {
  const [plan, setPlan] = useState<any>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function preview(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const d = new FormData(e.currentTarget);
    setBusy(true);
    setError("");
    try {
      setPlan(
        await api("register/preview", "POST", {
          path: d.get("path"),
          name: d.get("name"),
          version_label: d.get("version_label") || null,
          message: d.get("message") || null,
          alias: d.get("alias") || null,
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function confirm() {
    setBusy(true);
    setError("");
    try {
      await api("register/confirm", "POST", { plan_id: plan.plan_id });
      done();
    } catch (e) {
      setPlan(undefined);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog title="注册 Skill" close={close}>
      <form onSubmit={preview} onChange={() => setPlan(undefined)}>
        <fieldset disabled={busy}>
          <label>
            Skill 目录或 SKILL.md
            <input name="path" required placeholder="/path/to/skill" />
          </label>
          <label>
            名称
            <input name="name" required />
          </label>
          <label>
            初始版本号
            <input
              name="version_label"
              pattern={versionPattern}
              placeholder="v1.0.0"
              title="格式如 v1.2.0，不允许前导零"
            />
          </label>
          <label>
            注册说明
            <input name="message" />
          </label>
          <div className="fields">
            <label>
              别名
              <input name="alias" />
            </label>
          </div>
          <div className="actions">
            <button>预览注册内容</button>
          </div>
        </fieldset>
      </form>
      {plan && (
        <section>
          <h2>{plan.name}</h2>
          <code>{plan.path}</code>
          <p>将 Skill 保存到本地工作区，无需连接云端</p>
          <code>{plan.hash}</code>
          <div className="actions">
            <button disabled={busy} className="primary" onClick={confirm}>
              确认注册
            </button>
          </div>
        </section>
      )}
      <Message text={error} error />
    </Dialog>
  );
}
export function Export({
  version,
  versionLabel,
  close,
}: {
  version: Ref;
  versionLabel: string;
  close: () => void;
}) {
  const [directory, setDirectory] = useState("");
  const [plan, setPlan] = useState<any>();
  const [result, setResult] = useState<any>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function preview(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      setPlan(await api("export/preview", "POST", { version, directory }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function confirm() {
    setBusy(true);
    setError("");
    try {
      setResult(await api("export/confirm", "POST", { plan_id: plan.plan_id }));
      setPlan(undefined);
    } catch (e) {
      setPlan(undefined);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog title="导出版本" close={close}>
      <code>{versionLabel}</code>
      {!result && (
        <form onSubmit={preview}>
          <label>
            目标 Skill 目录
            <input
              required
              value={directory}
              onChange={(e) => {
                setDirectory(e.target.value);
                setPlan(undefined);
              }}
              placeholder="/path/to/exported-skill"
            />
          </label>
          <p className="muted">导出此版本的 SKILL.md 和全部辅助文件。</p>
          <div className="actions">
            <button disabled={busy}>预览写入计划</button>
          </div>
        </form>
      )}
      {plan && (
        <section>
          <code>{plan.directory}</code>
          <p>
            {plan.overwrite
              ? "目标目录将被完整替换，其中不属于此版本的文件将被移除。"
              : "将在目标目录写入完整快照。"}
          </p>
          <Changes text={plan.diff} />
          <div className="actions">
            <button className="primary" disabled={busy} onClick={confirm}>
              确认导出
            </button>
          </div>
        </section>
      )}
      {result && (
        <section>
          <h2>已导出</h2>
          <code>{result.path}</code>
          <p className="muted">由 Agent 在下次加载时读取。</p>
        </section>
      )}
      <Message text={error} error />
    </Dialog>
  );
}
export function SwitchVersion({
  skillId,
  versionId,
  versionLabel,
  close,
}: {
  skillId: string;
  versionId: string;
  versionLabel: string;
  close: () => void;
}) {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  async function apply() {
    setBusy(true);
    setError("");
    try {
      const plan = await api<{ plan_id: string }>("switch/preview", "POST", {
        skill_id: skillId,
        version_id: versionId,
      });
      await api("switch/confirm", "POST", { plan_id: plan.plan_id });
      setDone(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog title="设为当前版本" close={close}>
      <Message text={error} error />
      {done ? (
        <Message text="工作区当前版本已更新。" />
      ) : (
        <>
          <p>
            将版本 {versionLabel} 设为当前版本。需要写入目录时，请使用导出版本。
          </p>
          <div className="actions">
            <button disabled={busy} className="primary" onClick={apply}>
              确认切换
            </button>
          </div>
        </>
      )}
    </Dialog>
  );
}

export function Commit({
  skillId,
  close,
  done,
}: {
  skillId: string;
  close: () => void;
  done: (id: string) => void;
}) {
  const [mode, setMode] = useState("patch");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const current = useLoad<{ skill: { current_version_id: string } }>(
    () => api("skills/" + encodeURIComponent(skillId)),
    skillId,
  );
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const data = new FormData(e.currentTarget);
    setBusy(true);
    setError("");
    try {
      const result = await api<{ version: { version_id: string } }>(
        "commit",
        "POST",
        {
          skill_id: skillId,
          path: data.get("path"),
          message: data.get("message") || null,
          version_label: mode === "manual" ? data.get("version_label") : null,
          version_bump: mode === "manual" ? "patch" : mode,
          base_version_id: current.value!.skill.current_version_id,
        },
      );
      done(result.version.version_id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog title="提交新版本" close={close}>
      <form onSubmit={submit}>
        <label>
          源目录或 SKILL.md
          <input name="path" required />
        </label>
        <label>
          版本号方式
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="patch">修订号 +1（v1.2.0 → v1.2.1）</option>
            <option value="minor">次版本 +1（v1.2.0 → v1.3.0）</option>
            <option value="major">主版本 +1（v1.2.0 → v2.0.0）</option>
            <option value="manual">手动指定</option>
          </select>
        </label>
        {mode === "manual" && (
          <label>
            版本号
            <input
              name="version_label"
              required
              pattern={versionPattern}
              placeholder="v1.2.0"
              title="格式如 v1.2.0，不允许前导零"
            />
          </label>
        )}
        <label>
          提交说明
          <input name="message" />
        </label>
        <p>保存完整内容与辅助文件，并将新版本设为当前版本。</p>
        <button disabled={busy || !current.value}>提交</button>
      </form>
      <Message text={error || current.error} error />
    </Dialog>
  );
}
