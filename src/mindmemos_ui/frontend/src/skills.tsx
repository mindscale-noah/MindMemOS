import { useState } from "react";
import { api, query, versionLabel, type Selection, type Skill } from "./api";
import { Empty, Message, useLoad, useRefresh } from "./components";
import { Register } from "./operations";
type SkillPage = {
  items: Skill[];
  next_cursor: string | null;
  has_more: boolean;
};
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
  const [cursors, setCursors] = useState<(string | null)[]>([null]);
  const cursor = cursors[cursors.length - 1];
  const refresh = useRefresh(() => setReload((current) => current + 1));
  const { value, error, loading } = useLoad<SkillPage>(
    () =>
      api(
        `skills?${query({ page_size: "20", ...(cursor ? { cursor } : {}) })}`,
      ),
    `${cursor || "first"}:${reload}`,
    active,
  );
  const rows = (value?.items || []).filter((s) =>
    (s.name + " " + (s.alias || ""))
      .toLowerCase()
      .includes(search.toLowerCase()),
  );
  return (
    <>
      <div className="toolbar">
        <input
          aria-label="Search Skills on this page"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search this page by name or alias"
        />
        <div className="toolbar-actions">
          <button onClick={refresh.refresh}>Refresh</button>
          <button className="primary" onClick={() => setRegister(true)}>
            ＋ Register Skill
          </button>
        </div>
      </div>
      <Message text={refresh.error || error} error />
      {loading ? (
        <Empty>Loading…</Empty>
      ) : rows.length ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Skill</th>
                <th>Alias</th>
                <th>Active version</th>
                <th title="Most recently created version in the central repository">
                  Latest version
                </th>
                <th>Registered</th>
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
                    <span className="version-label" title={s.active_version_id}>
                      {versionLabel(
                        {
                          version_label: s.active_version_label,
                          implementation: s.implementation,
                        },
                        "No version label",
                      )}
                    </span>
                  </td>
                  <td>
                    <span className="version-label" title={s.latest_version_id}>
                      {versionLabel(
                        {
                          version_label: s.latest_version_label,
                          implementation: s.latest_implementation,
                        },
                        "No version label",
                      )}
                    </span>
                  </td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    {s.created_at && !Number.isNaN(Date.parse(s.created_at))
                      ? new Date(s.created_at).toLocaleString("en-US", {
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
        !error && <Empty>{search ? "No matching Skills" : "No Skills"}</Empty>
      )}
      {value && !loading && (
        <div className="memory-pagination">
          <span className="muted">Page {cursors.length}</span>
          <button
            type="button"
            disabled={cursors.length === 1}
            onClick={() => {
              setCursors((current) => current.slice(0, -1));
              setSearch("");
            }}
          >
            Previous
          </button>
          <button
            type="button"
            disabled={!value.has_more || !value.next_cursor}
            onClick={() => {
              setCursors((current) => [...current, value.next_cursor]);
              setSearch("");
            }}
          >
            Next
          </button>
        </div>
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
