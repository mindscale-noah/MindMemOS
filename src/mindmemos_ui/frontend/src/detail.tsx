import { useState, useEffect, type ReactNode } from "react";
import { api, query, ref, type Version, type Selection, type Ref } from "./api";
import { useLoad, Message, Empty, Dialog, VersionOptions } from "./components";
import { Export, SwitchVersion, Commit } from "./operations";
import {
  BundleBrowser,
  type BundleContent,
  type BundleComparison,
} from "./bundle";
export function Compare({
  selection,
  versions,
  selected,
  renderActions,
}: {
  selection: Selection;
  versions: Version[];
  selected: string;
  renderActions: (versionId: string) => ReactNode;
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
          {renderActions(before)}
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
          {renderActions(after)}
        </div>
      </div>
      <Message text={error} error />
      {busy && <Empty>正在对比…</Empty>}
      {result && (
        <BundleBrowser
          key={before + after}
          content={result.after}
          comparison={result}
          beforeLabel={
            versions.find((v) => v.version_id === before)?.version_label ||
            before
          }
          afterLabel={
            versions.find((v) => v.version_id === after)?.version_label || after
          }
        />
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
    <>
      <Message text={data.error} error />
      {data.loading ? (
        <Empty>加载中…</Empty>
      ) : (
        data.value && (
          <BundleBrowser key={JSON.stringify(version)} content={data.value} />
        )
      )}
    </>
  );
}
export function Detail({
  selection,
  back,
}: {
  selection: Selection;
  back: () => void;
}) {
  const [tab, setTab] = useState(selection.version ? "content" : "versions");
  const [switching, setSwitchVersion] = useState<Version>();
  const [version, setVersion] = useState(selection.version || "");
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
    skill: { current_version_id: string };
    version: Version;
  }>(
    () => api("skills/" + encodeURIComponent(selection.id)),
    selection.id + revision,
  );
  useEffect(() => {
    if (!version && data.value?.length)
      setVersion(data.value[data.value.length - 1].version_id);
  }, [data.value]);
  function renderActions(versionId: string) {
    const target = data.value?.find((v) => v.version_id === versionId);
    if (!target) return null;
    const label = target.version_label || target.version_id;
    const isCurrent = current.value?.skill.current_version_id === versionId;
    return (
      <div className="actions version-actions">
        {selection.local && (
          <button disabled={isCurrent} onClick={() => setSwitchVersion(target)}>
            {isCurrent ? `${label} · 当前版本` : `将 ${label} 设为当前版本`}
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
          {selection.local && (
            <button onClick={() => setCommitting(true)}>提交新版本</button>
          )}
        </div>
      </div>
      <p className="muted">
        当前版本：{current.value?.version.version_label || "—"}
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
              <table>
                <thead>
                  <tr>
                    <th>版本</th>
                    <th>状态</th>
                    <th>来源</th>
                    <th>创建时间</th>
                    <th>版本操作</th>
                  </tr>
                </thead>
                <tbody>
                  {[...data.value].reverse().map((v) => (
                    <tr key={v.version_id}>
                      <td>
                        <button
                          className="link"
                          onClick={() => {
                            setVersion(v.version_id);
                            setTab("content");
                          }}
                        >
                          {v.version_label || "未命名版本"}
                        </button>
                        <details className="muted">
                          <summary>审计信息</summary>
                          <code>{v.version_id}</code>
                          <button
                            className="link"
                            onClick={() =>
                              navigator.clipboard.writeText(v.version_id)
                            }
                          >
                            复制 ID
                          </button>
                        </details>
                      </td>
                      <td>
                        <span className="badge">{v.status}</span>
                      </td>
                      <td>{v.origin}</td>
                      <td>{new Date(v.created_at).toLocaleString()}</td>
                      <td>{renderActions(v.version_id)}</td>
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
              renderActions={renderActions}
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
      {switching && selection.local && (
        <SwitchVersion
          skillId={selection.local.skill_id}
          versionId={switching.version_id}
          versionLabel={switching.version_label || switching.version_id}
          close={() => {
            setSwitchVersion(undefined);
            setRevision(revision + 1);
          }}
        />
      )}{" "}
      {exporting && (
        <Export
          version={ref(selection, exporting.version_id)}
          versionLabel={exporting.version_label || exporting.version_id}
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
