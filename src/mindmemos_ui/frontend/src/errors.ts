import type { ConnectionStatus } from "./api";

export type Service = "memory" | "skill";
export const rateLimitMessage = "Too many requests. Please try again shortly.";

export function connectionMessage(service: Service, status: ConnectionStatus) {
  const name = service === "memory" ? "Memory" : "Skill";
  switch (status) {
    case "connected":
      return "";
    case "unconfigured":
      return `The ${name} service is not configured. Open Settings to configure its connection and API key.`;
    case "unauthorized":
      return `Authentication failed for the ${name} service. Check its API key in Settings.`;
    case "unavailable":
      return `Cannot connect to the ${name} service. Check its address in Settings and make sure the service is running, then retry.`;
  }
}

export function requestErrorMessage(
  path: string,
  status: number,
  detail?: unknown,
) {
  const service = /^(memories)(\/|\?|$)/.test(path)
    ? "memory"
    : /^(skills|graphs)(\/|\?|$)/.test(path)
      ? "skill"
      : undefined;
  const text = typeof detail === "string" ? detail.trim() : "";
  if (status === 429) return rateLimitMessage;
  if (status === 401)
    return service
      ? connectionMessage(service, "unauthorized")
      : "Authentication failed. Check your API key in Settings.";
  if (status === 403)
    return "Access denied or session expired. Check your permissions and refresh the page.";
  if (status === 503 && service && /configure.*api key/i.test(text)) {
    return connectionMessage(service, "unconfigured");
  }
  if ([502, 503, 504].includes(status))
    return service
      ? connectionMessage(service, "unavailable")
      : "The requested service is unavailable. Please retry shortly.";
  if (status >= 500)
    return "The request could not be completed because of a server error. Please retry shortly.";
  if (status === 409)
    return (
      text ||
      (/^(service-config|configurations|client-configurations)/.test(path)
        ? "The configuration file has changed. Copy your draft, then reload before saving."
        : "The resource has changed or conflicts with existing data. Refresh and try again.")
    );
  if (status === 422 || status === 400)
    return (
      text || "Some request fields are invalid. Check your input and try again."
    );
  if (status === 404)
    return (
      text || "The requested resource was not found. Refresh and try again."
    );
  return text || `Request failed (${status}). Please try again.`;
}
