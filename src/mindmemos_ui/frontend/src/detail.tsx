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
            基准版本
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
            目标版本
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
      {busy && <Empty>正在对比…</Empty>}
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
      labels={["当前查看版本"]}
    >
      <Message text={data.error} error />
      {data.loading ? (
        <Empty>加载中…</Empty>
      ) : (
        data.value && <BundleBrowser content={data.value} />
      )}
    </DynamicSkill>
  );
}
export function Detail({
  selection,
  back,
  tab,
  change,
}: {
  selection: Selection;
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
            {isActive ? `${label} · 生效版本` : `将 ${label} 设为生效版本`}
          </button>
        )}
        <button onClick={() => setExporting(target)}>导出 {label}</button>
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
      <button className="link breadcrumb" onClick={back}>
        ← 返回
      </button>
      <div className="row">
        <h1>{selection.name}</h1>
        <div className="actions">
          {selection.source !== "graph" && (
            <button onClick={() => setRemoving(true)}>删除 Skill</button>
          )}
          {selection.source === "local" && (
            <button onClick={() => setCommitting(true)}>提交新版本</button>
          )}
        </div>
      </div>
      <p className="muted">
        生效版本：{versionLabel(current.value?.version, "—")}
      </p>
      <Message text={current.error} error />
      <div className="tabs">
        {[
          ["content", "内容"],
          ["versions", "版本历史"],
          ["diff", "对比"],
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
        <Empty>加载中…</Empty>
      ) : !data.value?.length ? (
        <Empty>暂无可读取的版本</Empty>
      ) : (
        <>
          {tab === "versions" ? (
            <div className="table-wrap">
              <table className="version-history-table">
                <thead>
                  <tr>
                    <th>版本名</th>
                    <th>版本ID</th>
                    <th>来源</th>
                    <th>创建时间</th>
                    <th>提交信息</th>
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
                      <td>{new Date(v.created_at).toLocaleString()}</td>
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
              <div className="version-toolbar">
                <label className="version-picker">
                  版本
                  <select
                    value={version}
                    onChange={(e) => setVersion(e.target.value)}
                  >
                    <VersionOptions versions={data.value} />
                  </select>
                </label>
                {renderActions(version)}
              </div>
              {version && <Content version={ref(selection, version)} />}
            </>
          ) : (
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
        <Dialog title="删除 Skill" close={() => setRemoving(false)}>
          <p>
            <strong>{selection.name}</strong>{" "}
            的中心仓记录及版本历史将被删除，原导入和导出文件保留。有引用的 Skill
            无法删除。
          </p>
          <div className="actions">
            <button onClick={() => setRemoving(false)}>取消</button>
            <button disabled={busy} onClick={remove}>
              确认删除
            </button>
          </div>
          <Message text={error} error />
        </Dialog>
      )}
    </>
  );
}
