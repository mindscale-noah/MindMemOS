import { useRef, type ReactNode } from "react";
import type { Connections } from "./api";
import { Empty, Message } from "./components";
import { connectionMessage, type Service } from "./errors";

export function ServiceBoundary({
  service,
  connections,
  checkFailed,
  retry,
  settings,
  children,
}: {
  service: Service;
  connections: Connections | null;
  checkFailed: boolean;
  retry: () => void;
  settings: () => void;
  children: ReactNode;
}) {
  const status = connections?.[service].status;
  const hasConnected = useRef(false);
  if (status === "connected") hasConnected.current = true;
  if (!status && !checkFailed)
    return <Empty>Checking service connection…</Empty>;
  return (
    <>
      <div hidden={status === "connected"} className="service-error">
        <Message
          error
          text={
            status
              ? connectionMessage(service, status)
              : "Could not check service connections. Make sure the UI server is running, then retry."
          }
        />
        <div className="service-error-actions">
          <button onClick={retry}>Retry</button>
          <button onClick={settings}>Open Settings</button>
        </div>
      </div>
      {hasConnected.current && (
        <div hidden={status !== "connected"}>{children}</div>
      )}
    </>
  );
}
