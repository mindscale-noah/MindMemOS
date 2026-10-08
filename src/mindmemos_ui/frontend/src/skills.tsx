import { useState } from "react";
import {
  api,
  query,
  versionLabel,
  type Selection,
  type Skill,
  type Version,
} from "./api";
import { Empty, Message, useLoad } from "./components";
import { Register } from "./operations";
export function Skills({
  open,
  active,
}: {
  open: (s: Selection) => void;
  active: boolean;
}) {
  const [search, setSearch] = useState("");
  const [register, setRegister] = useState(false);
  const [reload, setReload] = useState(0);
  const { value, error, loading } = useLoad<Skill[]>(
    async () => {
      const skills = await api<Skill[]>("skills");
      return Promise.all(
        skills.map(async (skill) => {
          if (skill.active_version_label && skill.latest_version_id)
            return skill;
          const history = await api<Version[]>(
            `versions?${query({ skill_id: skill.skill_id })}`,
          );
          const current = history.find(
            (v) => v.version_id === skill.active_version_id,
          );
          const latest = history.reduce<Version | undefined>(
            (best, version) =>
              !best ||
              Date.parse(version.created_at) > Date.parse(best.created_at)
                ? version
                : best,
            undefined,
          );
          return {
            ...skill,
            active_version_label:
              current?.version_label ?? skill.active_version_label,
            latest_version_id: latest?.version_id,
            latest_version_label: latest?.version_label,
            implementation: current?.implementation ?? skill.implementation,
            latest_implementation: latest?.implementation,
          };
        }),
      );
    },
    String(reload) + active,
  );
  const rows = (value || []).filter((s) =>
    (s.name + " " + (s.alias || ""))
      .toLowerCase()
      .includes(search.toLowerCase()),
  );
  return (
    <>
      <div className="row">
        <h1>Skill 服务</h1>
        <button className="primary" onClick={() => setRegister(true)}>
          ＋ 注册 Skill
        </button>
      </div>
      <p className="muted">
        管理当前 SDK 配置的 Skill 服务中的快照与版本。
      </p>
      <div className="toolbar">
        <input
          aria-label="搜索 Skill"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索名称或别名"
        />
        <button onClick={() => setReload(reload + 1)}>刷新</button>
      </div>
      <Message text={error} error />
      {loading ? (
        <Empty>加载中…</Empty>
      ) : rows.length ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Skill</th>
                <th>别名</th>
                <th>生效版本</th>
                <th title="中心仓最近创建的版本">最新版本</th>
                <th>注册时间</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((s) => (
                <tr key={s.skill_id}>
                  <td>
                    <button
                      className="link"
                      onClick={() =>
                        open({
                          source: "local",
                          id: s.skill_id,
                          name: s.name,
                          local: s,
                          version: s.active_version_id,
                        })
                      }
                    >
                      {s.name}
                    </button>
                  </td>
                  <td>{s.alias || "—"}</td>
                  <td>
                    <code title={s.active_version_id}>
                      {versionLabel(
                        {
                          version_label: s.active_version_label,
                          implementation: s.implementation,
                        },
                        "未设置版本号",
                      )}
                    </code>
                  </td>
                  <td>
                    <code title={s.latest_version_id}>
                      {versionLabel(
                        {
                          version_label: s.latest_version_label,
                          implementation: s.latest_implementation,
                        },
                        "未设置版本号",
                      )}
                    </code>
                  </td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    {s.created_at && !Number.isNaN(Date.parse(s.created_at))
                      ? new Date(s.created_at).toLocaleString("zh-CN", {
                          hour12: false,
                        })
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        !error && <Empty>{search ? "没有匹配的 Skill" : "暂无 Skill"}</Empty>
      )}
      {register && (
        <Register
          close={() => setRegister(false)}
          done={() => {
            setRegister(false);
            setReload(reload + 1);
          }}
        />
      )}
    </>
  );
}
