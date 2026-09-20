import { useState } from "react";
import { api, type Selection, type Skill } from "./api";
import { useLoad, Message, Empty } from "./components";
import { Register } from "./operations";
export function Skills({ open }: { open: (s: Selection) => void }) {
  const [search, setSearch] = useState("");
  const [register, setRegister] = useState(false);
  const [reload, setReload] = useState(0);
  const { value, error, loading } = useLoad<Skill[]>(
    () => api("skills"),
    String(reload),
  );
  const rows = (value || []).filter((s) =>
    (s.name + " " + (s.alias || ""))
      .toLowerCase()
      .includes(search.toLowerCase()),
  );
  return (
    <>
      <div className="row">
        <h1>Skills 中心仓</h1>
        <button className="primary" onClick={() => setRegister(true)}>
          ＋ 注册 Skill
        </button>
      </div>
      <p className="muted">
        统一管理 Skill 快照与版本。导入目录仅作为来源记录。
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
                <th>当前版本</th>
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
                          version: s.current_version_id,
                        })
                      }
                    >
                      {s.name}
                    </button>
                  </td>
                  <td>{s.alias || "—"}</td>
                  <td>
                    <code>{s.current_version_label || "未命名版本"}</code>
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
