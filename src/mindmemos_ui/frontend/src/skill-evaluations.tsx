import { useState } from "react";
import { api, type Ref, type Version } from "./api";
import { Dialog, Empty, Message, VersionOptions, useLoad } from "./components";

type Evaluation = {
  evaluation_id: string;
  algorithm: string;
  algorithm_version: string;
  prompt_version: string;
  summary: string | null;
  score_mean: number | null;
  scores: Record<string, unknown>;
  findings: {
    category: string;
    severity: string;
    message: string;
    evidence: string[];
  }[];
  artifacts: Record<string, string>;
  metrics: Record<string, unknown>;
  metadata: Record<string, unknown>;
  updated_at: string;
};

type EvaluationHistory = {
  items: Evaluation[];
  limit_reached: boolean;
};

const dateTime = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "medium",
});

function formatScore(score: unknown, algorithm: string) {
  return typeof score === "number"
    ? `${Number(score.toFixed(2))}${algorithm === "skill_judge" ? " / 5" : ""}`
    : "—";
}

function EvaluationDetails({
  record,
  close,
}: {
  record: Evaluation;
  close: () => void;
}) {
  const reasons = record.metrics.score_reasons as
    | Record<string, string>
    | undefined;
  return (
    <Dialog title="Evaluation details" close={close}>
      <div className="evaluation-details">
        <p className="muted">
          <time dateTime={record.updated_at}>
            {dateTime.format(new Date(record.updated_at))}
          </time>
          {" · "}
          <strong>{formatScore(record.score_mean, record.algorithm)}</strong>
        </p>
        {record.summary && <p>{record.summary}</p>}
        {Object.keys(record.scores).length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Dimension</th>
                  <th>Score</th>
                  {reasons && <th>Reason</th>}
                </tr>
              </thead>
              <tbody>
                {Object.entries(record.scores).map(([dimension, score]) => (
                  <tr key={dimension}>
                    <td>{dimension.replaceAll("_", " ")}</td>
                    <td className="evaluation-score">
                      {formatScore(score, record.algorithm)}
                    </td>
                    {reasons && <td>{reasons[dimension] || "—"}</td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {record.findings.length > 0 && (
          <section>
            <h3>Findings</h3>
            <ul className="evaluation-findings">
              {record.findings.map((finding, index) => (
                <li key={index}>
                  <span className="badge">{finding.severity}</span>{" "}
                  {finding.message}
                  {finding.evidence.length > 0 && (
                    <ul>
                      {finding.evidence.map((evidence, index) => (
                        <li key={index}>{evidence}</li>
                      ))}
                    </ul>
                  )}
                </li>
              ))}
            </ul>
          </section>
        )}
        {Object.entries(record.artifacts).map(([name, content]) => (
          <details key={name}>
            <summary>{name}</summary>
            <pre>{content}</pre>
          </details>
        ))}
        <details>
          <summary>Run details</summary>
          <pre>
            {JSON.stringify(
              {
                evaluation_id: record.evaluation_id,
                algorithm: record.algorithm,
                algorithm_version: record.algorithm_version,
                prompt_version: record.prompt_version,
                metrics: record.metrics,
                metadata: record.metadata,
              },
              null,
              2,
            )}
          </pre>
        </details>
      </div>
    </Dialog>
  );
}

export function SkillEvaluations({
  target,
  versions,
  setVersion,
}: {
  target: Ref;
  versions: Version[];
  setVersion: (version: string) => void;
}) {
  const { skill_id: skillId, version_id: version } = target;
  const path = `skills/${encodeURIComponent(skillId)}/versions/${encodeURIComponent(version)}/evaluations`;
  const [revision, setRevision] = useState(0);
  const history = useLoad(
    () => api<EvaluationHistory>(path),
    `${path}:${revision}`,
  );
  const [selected, setSelected] = useState<Evaluation>();
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState("");

  async function runJudge() {
    setRunning(true);
    setRunError("");
    try {
      await api("skills/evaluations/run", "POST", target);
      setRevision((previous) => previous + 1);
    } catch (cause) {
      setRunError((cause as Error).message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <>
      <div className="version-toolbar">
        <label className="version-picker">
          Version
          <select
            value={version}
            onChange={(event) => setVersion(event.target.value)}
            disabled={running}
          >
            <VersionOptions versions={versions} />
          </select>
        </label>
        <button
          type="button"
          className="primary"
          onClick={runJudge}
          disabled={
            running || !versions.some((item) => item.version_id === version)
          }
        >
          {running ? "Running Skill Judge…" : "Run Skill Judge"}
        </button>
      </div>
      <Message text={runError} error />
      {history.loading ? (
        <Empty>Loading evaluations…</Empty>
      ) : history.error ? (
        <Message text={history.error} error />
      ) : !history.value?.items.length ? (
        <Empty>No completed evaluations for this version</Empty>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Evaluated</th>
                <th>Score</th>
                <th>Summary</th>
              </tr>
            </thead>
            <tbody>
              {history.value.items.map((record) => (
                <tr
                  key={record.evaluation_id}
                  className="evaluation-row"
                  onClick={() => setSelected(record)}
                >
                  <td>
                    <button type="button" className="link evaluation-time">
                      <time dateTime={record.updated_at}>
                        {dateTime.format(new Date(record.updated_at))}
                      </time>
                    </button>
                  </td>
                  <td className="evaluation-score">
                    {formatScore(record.score_mean, record.algorithm)}
                  </td>
                  <td>
                    <div className="evaluation-summary">
                      {record.summary || "—"}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {history.value?.limit_reached && (
        <p className="muted">Showing the latest 1,000 evaluations.</p>
      )}
      {selected && (
        <EvaluationDetails
          record={selected}
          close={() => setSelected(undefined)}
        />
      )}
    </>
  );
}
