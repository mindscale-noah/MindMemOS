import { useMemo, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import "highlight.js/styles/github.css";
import { Dialog } from "./components";

export function splitFrontmatter(text: string) {
  const match =
    /^\uFEFF?---[ \t]*\r?\n([\s\S]*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|$)/.exec(
      text,
    );
  return match
    ? { metadata: match[1], body: text.slice(match[0].length) }
    : { metadata: undefined, body: text };
}

export function resolveBundleLink(path: string, href: string, paths: string[]) {
  if (
    !href ||
    /^[a-z][a-z\d+.-]*:/i.test(href) ||
    href.startsWith("/") ||
    href.startsWith("#")
  )
    return undefined;
  try {
    const target = decodeURIComponent(href.split(/[?#]/)[0]);
    if (target.includes("\\") || target.startsWith("/")) return undefined;
    const parts = path.split("/").slice(0, -1);
    for (const part of target.split("/")) {
      if (!part || part === ".") continue;
      if (part === "..") {
        if (!parts.length) return undefined;
        parts.pop();
      } else parts.push(part);
    }
    const resolved = parts.join("/");
    return paths.includes(resolved) ? resolved : undefined;
  } catch {
    return undefined;
  }
}

hljs.registerLanguage("python", python);

function PythonCode({ text }: { text: string }) {
  const html = useMemo(
    () => hljs.highlight(text, { language: "python" }).value,
    [text],
  );
  // highlight.js escapes source text before adding its own token markup.
  return (
    <code
      className="hljs language-python"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function FilePreview({
  path,
  text,
  paths,
  onOpenFile,
}: {
  path: string;
  text: string;
  paths: string[];
  onOpenFile: (path: string) => void;
}) {
  const [external, setExternal] = useState<string>();
  const extension = path.split(".").pop()?.toLowerCase();
  if (extension === "md" || extension === "markdown") {
    const { metadata, body } = splitFrontmatter(text);
    return (
      <article className="markdown-preview">
        {metadata !== undefined && (
          <details className="frontmatter">
            <summary>File metadata</summary>
            <pre>
              <code>{metadata}</code>
            </pre>
          </details>
        )}
        <Markdown
          remarkPlugins={[remarkGfm]}
          components={{
            a({ href = "", children }) {
              const target = resolveBundleLink(path, href, paths);
              if (target)
                return (
                  <button
                    className="markdown-link"
                    title={`Open ${target} in the active version`}
                    onClick={() => onOpenFile(target)}
                  >
                    {children}
                  </button>
                );
              if (/^https?:\/\//i.test(href))
                return (
                  <button
                    className="markdown-link"
                    title={`View external link: ${href}`}
                    onClick={() => setExternal(href)}
                  >
                    {children} ↗
                  </button>
                );
              return (
                <span
                  className="unavailable-link"
                  title={`Cannot open in the active version: ${href || "Invalid link"}`}
                >
                  {children} (link unavailable)
                </span>
              );
            },
            img({ alt }) {
              return (
                <span className="muted">
                  [Image: {alt || "Untitled"}; not loaded]
                </span>
              );
            },
            code({ className, children, node: _node, ...props }) {
              if (/^language-(python|py)$/.test(className || "")) {
                return <PythonCode text={String(children)} />;
              }
              return (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            },
          }}
        >
          {body}
        </Markdown>
        {external && (
          <Dialog title="External link" close={() => setExternal(undefined)}>
            <p>The following URL will open in a new tab:</p>
            <p className="external-url">{external}</p>
            <div className="actions">
              <button onClick={() => setExternal(undefined)}>Cancel</button>
              <a
                href={external}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setExternal(undefined)}
              >
                Open link ↗
              </a>
            </div>
          </Dialog>
        )}
      </article>
    );
  }
  if (extension === "py" || extension === "pyw" || extension === "pyi") {
    return (
      <pre className="python-preview">
        <PythonCode text={text} />
      </pre>
    );
  }
  return <pre>{text}</pre>;
}
