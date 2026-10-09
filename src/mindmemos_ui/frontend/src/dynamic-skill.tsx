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
            Task
            <textarea
              aria-label="Assembly task"
              placeholder="Enter a task to assemble the Skill"
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
                ? "Assembling…"
                : versions.length === 2
                  ? "Assemble and compare for the same task"
                  : "Assemble for task"}
            </button>
            {assembled && (
              <button type="button" onClick={() => setAssembled(undefined)}>
                View original contents
              </button>
            )}
          </div>
          <Message text={error} error />
        </form>
      )}
      {busy && <Empty>Assembling Skill for the task…</Empty>}
      {assembled ? (
        <>
          <p className="muted">Assembly result: {assembled.task.instruction}</p>
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
