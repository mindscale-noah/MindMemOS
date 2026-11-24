import React, { useState } from "react";
import { api, versionPattern, type Ref } from "./api";
import { useLoad, Message, Dialog, Changes } from "./components";
async function chooseSkillDirectory(setPath: (path: string) => void) {
  const result = await api<{ path: string | null; unavailable: boolean }>(
    "skills/choose-directory", "POST", {},
  );
  if (result.unavailable) throw new Error("The local folder picker is unavailable. Enter the path manually.");
  if (result.path) setPath(result.path);
}
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
  const [path, setPath] = useState("");
  async function pickDirectory() {
    try {
      await chooseSkillDirectory((selected) => { setPath(selected); setPlan(undefined); });
    } catch (e) { setError((e as Error).message); }
  }
  async function preview(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const d = new FormData(e.currentTarget);
    setBusy(true);
    setError("");
    try {
      setPlan(
        await api("register/preview", "POST", {
          path,
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
    <Dialog title="Register Skill" close={close}>
      <form onSubmit={preview} onChange={() => setPlan(undefined)}>
        <fieldset disabled={busy}>
          <label>
            Skill folder
            <div className="fields">
              <input value={path} onChange={(event) => setPath(event.target.value)} required placeholder="Choose a folder containing SKILL.md" />
              <button type="button" onClick={pickDirectory}>Choose folder</button>
            </div>
          </label>
          <label>
            Alias (optional)
            <input name="alias" placeholder="Defaults to the name in SKILL.md" />
          </label>
          <p className="muted">The Skill name is read from SKILL.md.</p>
          <label>
            Initial version
            <input
              name="version_label"
              pattern={versionPattern}
              placeholder="v1.0.0"
              title="Use a version such as v1.2.0, without leading zeros"
            />
          </label>
          <label>
            Commit message (optional)
            <input name="message" />
          </label>
          <div className="actions">
            <button>Preview registration</button>
          </div>
        </fieldset>
      </form>
      {plan && (
        <section>
          <h2>{plan.name}</h2>
          <code>{plan.path}</code>
          <p>Confirm to upload the complete contents to the configured Skill service.</p>
          <code>{plan.service_url}</code>
          <code>{plan.hash}</code>
          <div className="actions">
            <button disabled={busy} className="primary" onClick={confirm}>
              Confirm registration
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
  async function pickDirectory() {
    try {
      await chooseSkillDirectory((selected) => { setDirectory(selected); setPlan(undefined); });
    } catch (e) { setError((e as Error).message); }
  }
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
    <Dialog title="Export version" close={close}>
      <code>{versionLabel}</code>
      {!result && (
        <form onSubmit={preview}>
          <label>
            Export parent folder
            <div className="fields">
              <input required value={directory} onChange={(event) => { setDirectory(event.target.value); setPlan(undefined); }} placeholder="Choose a parent folder" />
              <button type="button" disabled={busy} onClick={pickDirectory}>Choose folder</button>
            </div>
          </label>
          <p className="muted">A folder named after this Skill will be created inside the selected folder, containing SKILL.md and all supporting files.</p>
          <div className="actions">
            <button disabled={busy}>Preview export</button>
          </div>
        </form>
      )}
      {plan && (
        <section>
          <p>Final Skill folder</p>
          <code>{plan.directory}</code>
          <p>
            {plan.overwrite
              ? "This Skill folder will be replaced, removing files outside this version. Other files in the parent folder will be kept."
              : "The complete snapshot will be written to this Skill folder. Other files in the parent folder will be kept."}
          </p>
          <Changes text={plan.diff} />
          <div className="actions">
            <button className="primary" disabled={busy} onClick={confirm}>
              Confirm export
            </button>
          </div>
        </section>
      )}
      {result && (
        <section>
          <h2>Exported</h2>
          <code>{result.path}</code>
          <p className="muted">Select this Skill folder to register the exported version again.</p>
        </section>
      )}
      <Message text={error} error />
    </Dialog>
  );
}
export function ActivateVersion({
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
      const plan = await api<{ plan_id: string }>("active/preview", "POST", {
        skill_id: skillId,
        version_id: versionId,
      });
      await api("active/confirm", "POST", { plan_id: plan.plan_id });
      setDone(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog title="Activate version" close={close}>
      <Message text={error} error />
      {done ? (
        <Message text="The active version has been updated in the Skill service." />
      ) : (
        <>
          <p>
            Make {versionLabel} the active version. Exports use the active version by default.
          </p>
          <div className="actions">
            <button disabled={busy} className="primary" onClick={apply}>
              Confirm activation
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
  const [path, setPath] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function pickDirectory() {
    setError("");
    try {
      await chooseSkillDirectory(setPath);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const current = useLoad<{ skill: { active_version_id: string } }>(
    () => api("skills/" + encodeURIComponent(skillId)),
    skillId,
  );
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const data = new FormData(e.currentTarget);
    const submitter = (e.nativeEvent as SubmitEvent).submitter as HTMLButtonElement | null;
    setBusy(true);
    setError("");
    try {
      const result = await api<{ version: { version_id: string } }>(
        "commit",
        "POST",
        {
          skill_id: skillId,
          path,
          message: data.get("message") || null,
          version_label: mode === "manual" ? data.get("version_label") : null,
          version_bump: mode === "manual" ? "patch" : mode,
          base_version_id: current.value!.skill.active_version_id,
          active: submitter?.value === "activate",
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
    <Dialog title="Commit new version" close={close}>
      <form onSubmit={submit}>
        <label>
          Skill folder
          <div className="fields folder-picker">
            <input
              name="path"
              value={path}
              onChange={(event) => setPath(event.target.value)}
              required
              disabled={busy}
              placeholder="Choose a folder containing SKILL.md"
            />
            <button type="button" disabled={busy} onClick={pickDirectory}>
              Choose folder
            </button>
          </div>
        </label>
        <label>
          Version numbering
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="patch">Patch +1 (v1.2.0 → v1.2.1)</option>
            <option value="minor">Minor +1 (v1.2.0 → v1.3.0)</option>
            <option value="major">Major +1 (v1.2.0 → v2.0.0)</option>
            <option value="manual">Manual</option>
          </select>
        </label>
        {mode === "manual" && (
          <label>
            Version
            <input
              name="version_label"
              required
              pattern={versionPattern}
              placeholder="v1.2.0"
              title="Use a version such as v1.2.0, without leading zeros"
            />
          </label>
        )}
        <label>
          Commit message
          <input name="message" />
        </label>
        <div className="actions">
          <button type="submit" value="commit" disabled={busy || !current.value}>Commit</button>
          <button type="submit" value="activate" className="primary" disabled={busy || !current.value}>Commit &amp; activate</button>
        </div>
      </form>
      <Message text={error || current.error} error />
    </Dialog>
  );
}
