import { useEffect, useRef, useState, type ReactNode } from "react";
import { api, query, type Ref } from "./api";
import { Empty, Message, useLoad } from "./components";
import {
  BundleBrowser,
  type BundleContent,
  type BundleComparison,
} from "./bundle";

type Implementation = { implementation: string; dynamic: boolean };
type Assembly = {
  contents: BundleContent[];
  task: { id: string; instruction: string };
  comparison?: Pick<BundleComparison, "files" | "implementation_changed">;
};

export function DynamicSkill({
  versions,
  labels,
  children,
}: {
  versions: Ref[];
  labels: string[];
  children: ReactNode;
}) {
  const metadata = useLoad<Implementation[]>(
    () =>
      Promise.all(
        versions.map((version) =>
          api<Implementation>("implementation?" + query(version)),
        ),
      ),
    JSON.stringify(versions),
  );
  const [instruction, setInstruction] = useState("");
  const [assembled, setAssembled] = useState<Assembly>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const dynamic = metadata.value?.some((info) => info.dynamic);
  async function assemble() {
    setBusy(true);
    setError("");
    setAssembled(undefined);
    try {
      const result = await api<Assembly>("assemble", "POST", {
        versions,
        instruction,
      });
      if (active.current) setAssembled(result);
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }
  return (
    <>
      <Message text={metadata.error} error />
      {dynamic && (
        <form
          className="skill-assembly"
          onSubmit={(event) => {
            event.preventDefault();
            void assemble();
          }}
        >
          <label>
            任务
            <textarea
              aria-label="组装任务"
              placeholder="输入任务，现场组装 Skill"
              value={instruction}
              disabled={busy}
              onChange={(event) => {
                setInstruction(event.target.value);
                setAssembled(undefined);
                setError("");
              }}
            />
          </label>
          <div className="actions">
            <button
              className="primary"
              disabled={busy || !instruction.trim()}
              type="submit"
            >
              {busy
                ? "正在组装…"
                : versions.length === 2
                  ? "按同一任务组装并对比"
                  : "按任务组装"}
            </button>
            {assembled && (
              <button type="button" onClick={() => setAssembled(undefined)}>
                查看原始内容
              </button>
            )}
          </div>
          <Message text={error} error />
        </form>
      )}
      {busy && <Empty>正在为任务组装 Skill…</Empty>}
      {assembled ? (
        <>
          <p className="muted">任务组装结果：{assembled.task.instruction}</p>
          <BundleBrowser
            key={assembled.task.id}
            content={assembled.contents[assembled.contents.length - 1]}
            comparison={
              assembled.comparison
                ? {
                    ...assembled.comparison,
                    before: assembled.contents[0],
                    after: assembled.contents[1],
                  }
                : undefined
            }
            beforeLabel={labels[0]}
            afterLabel={labels[1]}
          />
        </>
      ) : (
        children
      )}
    </>
  );
}
