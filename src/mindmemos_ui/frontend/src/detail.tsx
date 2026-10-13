import { Trajectories } from "./trajectories";
import { SkillEditor } from "./skill-editor";
import { DynamicSkill } from "./dynamic-skill";
import { useState, useEffect } from "react";
import {
  api,
  query,
  ref,
  versionLabel,
  type Version,
  type Selection,
  type Ref,
} from "./api";
import { useLoad, Message, Empty, Dialog, VersionOptions } from "./components";
import { Export, ActivateVersion, Commit } from "./operations";
import {
  BundleBrowser,
  type BundleContent,
  type BundleComparison,
} from "./bundle";
export function Compare({
  selection,
  versions,
  selected,
}: {
  selection: Selection;
  versions: Version[];
  selected: string;
}) {
  const [before, setBefore] = useState(versions[0]?.version_id || "");
  const [after, setAfter] = useState(selected);
  const [result, setResult] = useState<BundleComparison>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const { source, id } = selection;
  useEffect(() => {
    let active = true;
    setError("");
    setResult(undefined);
    if (!before || !after) {
      setBusy(false);
      return;
    }
    setBusy(true);
    async function compare() {
      try {
        const beforeRef: Ref = { source, skill_id: id, version_id: before };
        const afterRef: Ref = { source, skill_id: id, version_id: after };
        const [diff, left, right] = await Promise.all([
          api<Pick<BundleComparison, "files" | "implementation_changed">>(
            "diff",
            "POST",
            {
              before: beforeRef,
              after: afterRef,
            },
          ),
          api<BundleContent>("content?" + query(beforeRef)),
          api<BundleContent>("content?" + query(afterRef)),
        ]);
        if (active) setResult({ ...diff, before: left, after: right });
      } catch (e) {
        if (active) setError((e as Error).message);
      } finally {
        if (active) setBusy(false);
      }
    }
    void compare();
    return () => {
      active = false;
    };
  }, [source, id, before, after]);
  return (
    <>
      <div className="comparison-versions">
        <div className="version-card">
          <label>
            Base version
            <select
              value={before}
              onChange={(e) => {
                setBefore(e.target.value);
                setResult(undefined);
              }}
            >
              <VersionOptions versions={versions} />
            </select>
          </label>
        </div>
        <div className="version-card">
          <label>
            Target version
            <select
              value={after}
              onChange={(e) => {
                setAfter(e.target.value);
                setResult(undefined);
              }}
            >
              <VersionOptions versions={versions} />
            </select>
          </label>
        </div>
      </div>
      <Message text={error} error />
      {busy && <Empty>Comparing…</Empty>}
      {before && after && (
        <DynamicSkill
          key={source + id + before + after}
          versions={[
            { source, skill_id: id, version_id: before },
            { source, skill_id: id, version_id: after },
          ]}
          labels={[
            versionLabel(
              versions.find((v) => v.version_id === before),
              before,
            ),
            versionLabel(
              versions.find((v) => v.version_id === after),
              after,
            ),
          ]}
        >
          {result && (
            <BundleBrowser
              key={before + after}
              content={result.after}
              comparison={result}
              beforeLabel={versionLabel(
                versions.find((v) => v.version_id === before),
                before,
              )}
              afterLabel={versionLabel(
                versions.find((v) => v.version_id === after),
                after,
              )}
            />
          )}
        </DynamicSkill>
      )}
    </>
  );
}
export function Content({ version }: { version: Ref }) {
  const data = useLoad<BundleContent>(
    () => api("content?" + query(version)),
    JSON.stringify(version),
  );
  return (
    <DynamicSkill
      key={JSON.stringify(version)}
      versions={[version]}
      labels={["Selected version"]}
    >
      <Message text={data.error} error />
      {data.loading ? (
        <Empty>Loading…</Empty>
      ) : (
        data.value && <>
          {Object.entries(data.value.evaluations || {}).map(([phase, result]) => (
            <section key={phase} className="panel">
              <h3>{phase === "test" ? "Test evaluation" : "Validation evaluation"}</h3>
              <table><thead><tr><th>Tasks</th><th>Success rate</th><th>Execution errors</th><th>Total tokens</th><th>Elapsed seconds</th></tr></thead>
                <tbody><tr><td>{result.count}</td><td>{result.success_rate === null ? "Unknown" : `${(result.success_rate * 100).toFixed(2)}%`}</td>
                  <td>{result.errors}</td><td>{result.total_tokens?.toLocaleString() ?? "Unknown"}</td>
                  <td>{result.wall_seconds?.toFixed(2) ?? "Unknown"}</td></tr></tbody></table>
              <p>Input tokens: {result.input_tokens?.toLocaleString() ?? "Unknown"} · Output tokens: {result.output_tokens?.toLocaleString() ?? "Unknown"}</p>
              <p>Trajectory JSONL: <code>{result.trajectory_file}</code></p>
            </section>
          ))}
          <BundleBrowser content={data.value} />
        </>
      )}
    </DynamicSkill>
  );
}
export function Detail({
  selection,
  dirty,
  back,
  tab,
  change,
}: {
  selection: Selection;
  dirty: (value: boolean) => void;
  back: () => void;
  tab: string;
  change: (tab: string, version?: string, replace?: boolean) => void;
}) {
  const setTab = (next: string) => change(next, version);
  const [activating, setActivateVersion] = useState<Version>();
  const version = selection.version || "";
  const setVersion = (next: string) => change(tab, next);
  const [exporting, setExporting] = useState<Version>();
  const [removing, setRemoving] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    setEditing(false);
  }, [selection.version, tab]);
  const [committing, setCommitting] = useState(false);
  const [revision, setRevision] = useState(0);
  const data = useLoad<Version[]>(
    () =>
      selection.id
        ? api(
            "versions?" +
              query({ source: selection.source, skill_id: selection.id }),
          )
        : Promise.resolve([]),
    selection.source + selection.id + revision,
  );
  const current = useLoad<{
    skill: { active_version_id: string };
    version: Version;
  }>(
    () => api("skills/" + encodeURIComponent(selection.id)),
    selection.id + revision,
  );
  useEffect(() => {
    if (!version && data.value?.length)
      change(tab, data.value[data.value.length - 1].version_id, true);
  }, [data.value, version, tab]);
  function renderActions(versionId: string) {
    const target = data.value?.find((v) => v.version_id === versionId);
    if (!target) return null;
    const label = versionLabel(target, target.version_id);
    const isActive = current.value?.skill.active_version_id === versionId;
    return (
      <div className="actions version-actions">
        {selection.source === "local" && (
          <button
            disabled={isActive}
            onClick={() => setActivateVersion(target)}
          >
            {isActive ? `${label} · Active` : `Activate ${label}`}
          </button>
        )}
        <button onClick={() => setExporting(target)}>Export {label}</button>
      </div>
    );
  }
  async function remove() {
    setBusy(true);
    setError("");
    try {
      await api(
        selection.local
          ? "skills/" + encodeURIComponent(selection.local.skill_id)
          : "skills/" + encodeURIComponent(selection.id),
        "DELETE",
      );
      back();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="skill-heading">
        <button className="link breadcrumb" onClick={back}>
          ← Back
        </button>
        <div className="row skill-heading-main">
          <div className="skill-identity">
            <h1>{selection.name}</h1>
            <span className="muted">
              Active version: {versionLabel(current.value?.version, "—")}
            </span>
          </div>
          <div className="actions">
            {selection.source !== "graph" && !editing && (
              <details className="skill-more">
                <summary>More</summary>
                <div className="skill-more-menu">
                  <button onClick={() => setRemoving(true)}>
                    Delete Skill
                  </button>
                </div>
              </details>
            )}
            {selection.source === "local" && !editing && (
              <button onClick={() => setCommitting(true)}>
                Commit new version
              </button>
            )}
          </div>
        </div>
      </div>
      <Message text={current.error} error />
      <div className="tabs">
        {[
          ["content", "Content"],
          ["versions", "Version history"],
          ["diff", "Compare"],
          ...(selection.source === "local"
            ? [["trajectories", "Trajectories"]]
            : []),
        ].map(([key, label]) => (
          <button
            key={key}
            className={tab === key ? "active" : ""}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>
      <Message text={data.error} error />
      {data.loading ? (
        <Empty>Loading…</Empty>
      ) : !data.value?.length ? (
        <Empty>No readable versions</Empty>
      ) : (
        <>
          {selection.source === "local" && (
            <Trajectories
              active={tab === "trajectories"}
              skillId={selection.id}
              version={version}
              versions={data.value}
              setVersion={setVersion}
            />
          )}
          {tab === "versions" ? (
            <div className="table-wrap">
              <table className="version-history-table">
                <thead>
                  <tr>
                    <th>Version label</th>
                    <th>Version ID</th>
                    <th>Source</th>
                    <th>Created</th>
                    <th>Commit message</th>
                  </tr>
                </thead>
                <tbody>
                  {[...data.value].reverse().map((v) => (
                    <tr key={v.version_id}>
                      <td>
                        <button
                          className="link"
                          onClick={() => {
                            change("content", v.version_id);
                          }}
                        >
                          {versionLabel(v)}
                        </button>
                      </td>
                      <td>
                        <code>{v.version_id}</code>
                      </td>
                      <td>{v.origin}</td>
                      <td>
                        {new Date(v.created_at).toLocaleString("en-US", {
                          hour12: false,
                        })}
                      </td>
                      <td className="version-commit-message">
                        {v.commit_message || "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : tab === "content" ? (
            <>
              {editing ? (
                <SkillEditor
                  version={ref(selection, version)}
                  label={versionLabel(
                    data.value.find((item) => item.version_id === version),
                    version,
                  )}
                  dirty={dirty}
                  close={() => setEditing(false)}
                  done={(id) => {
                    setEditing(false);
                    setRevision((value) => value + 1);
                    change("content", id);
                  }}
                />
              ) : (
                <>
                  <div className="version-toolbar">
                    <label className="version-picker">
                      Version
                      <select
                        value={version}
                        onChange={(e) => setVersion(e.target.value)}
                      >
                        <VersionOptions versions={data.value} />
                      </select>
                    </label>
                    <div className="actions">
                      {selection.source === "local" && version && (
                        <button
                          className="primary"
                          onClick={() => setEditing(true)}
                        >
                          Edit
                        </button>
                      )}
                      {renderActions(version)}
                    </div>
                  </div>
                  {version && <Content version={ref(selection, version)} />}
                </>
              )}
            </>
          ) : tab === "trajectories" ? null : (
            <Compare
              selection={selection}
              versions={data.value}
              selected={version}
            />
          )}
        </>
      )}
      {committing && (
        <Commit
          skillId={selection.id}
          close={() => setCommitting(false)}
          done={(id) => {
            setVersion(id);
            setCommitting(false);
            setRevision(revision + 1);
          }}
        />
      )}
      {activating && selection.source === "local" && (
        <ActivateVersion
          skillId={selection.id}
          versionId={activating.version_id}
          versionLabel={versionLabel(activating, activating.version_id)}
          close={() => {
            setActivateVersion(undefined);
            setRevision(revision + 1);
          }}
        />
      )}{" "}
      {exporting && (
        <Export
          version={ref(selection, exporting.version_id)}
          versionLabel={versionLabel(exporting, exporting.version_id)}
          close={() => setExporting(undefined)}
        />
      )}{" "}
      {removing && (
        <Dialog title="Delete Skill" close={() => setRemoving(false)}>
          <p>
            <strong>{selection.name}</strong> will be removed from the central
            repository along with its version history. Imported and exported
            files will be preserved. Skills that are still referenced cannot be
            deleted.
          </p>
          <div className="actions">
            <button onClick={() => setRemoving(false)}>Cancel</button>
            <button disabled={busy} onClick={remove}>
              Confirm deletion
            </button>
          </div>
          <Message text={error} error />
        </Dialog>
      )}
    </>
  );
}
