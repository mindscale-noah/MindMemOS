import { rateLimitMessage } from "./errors";
import React, { useEffect, useRef, useState } from "react";
import { api, versionLabel, type Version } from "./api";
export function useRefresh(action: () => void) {
  const [error, setError] = useState("");
  const pending = useRef(false);
  async function refresh() {
    if (pending.current) {
      setError(rateLimitMessage);
      return;
    }
    pending.current = true;
    setError("");
    try {
      await api("refresh", "POST");
      action();
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      pending.current = false;
    }
  }
  return { refresh, error };
}
export function useLoad<T>(
  load: () => Promise<T>,
  key: string,
  enabled = true,
) {
  const [value, setValue] = useState<T>();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    if (!enabled) return;
    let active = true;
    setLoading(true);
    setError("");
    setValue(undefined);
    load()
      .then((v) => {
        if (active) setValue(v);
      })
      .catch((e) => {
        if (active) setError(e.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [key, enabled]);
  return { value, error, loading };
}
export function Message({
  text,
  error = false,
}: {
  text?: string;
  error?: boolean;
}) {
  return text ? (
    <p
      className={error ? "message error" : "message"}
      role={error ? "alert" : "status"}
    >
      {text}
    </p>
  ) : null;
}
export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="empty">{children}</div>;
}
export function VersionOptions({ versions }: { versions: Version[] }) {
  return (
    <>
      {versions.map((v) => (
        <option key={v.version_id} value={v.version_id}>
          {versionLabel(v)}
        </option>
      ))}
    </>
  );
}
export function Dialog({
  title,
  close,
  children,
}: {
  title: string;
  close: () => void;
  children: React.ReactNode;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  return (
    <dialog
      ref={dialog}
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
    >
      <div className="row">
        <h2>{title}</h2>
        <button type="button" onClick={close} aria-label="Close">
          ×
        </button>
      </div>
      {children}
    </dialog>
  );
}
export function Changes({ text }: { text: string }) {
  return (
    <pre className="diff">
      {text
        ? text.split("\n").map((line, i) => (
            <span
              key={i}
              className={
                line.startsWith("+")
                  ? "added"
                  : line.startsWith("-")
                    ? "removed"
                    : ""
              }
            >
              {line || " "}
              <br />
            </span>
          ))
        : "Unchanged"}
    </pre>
  );
}
