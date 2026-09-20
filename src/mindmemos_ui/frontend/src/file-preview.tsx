import { useMemo } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import "highlight.js/styles/github.css";

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

export function FilePreview({ path, text }: { path: string; text: string }) {
  const extension = path.split(".").pop()?.toLowerCase();
  if (extension === "md" || extension === "markdown") {
    return (
      <article className="markdown-preview">
        <Markdown
          remarkPlugins={[remarkGfm]}
          components={{
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
          {text}
        </Markdown>
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
