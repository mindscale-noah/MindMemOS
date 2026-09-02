import { spawn } from "node:child_process";

type SpawnResult = {
  stdout: string;
  stderr: string;
};

export async function spawnFileJson<T>(command: string, args: string[], stdin?: string): Promise<T> {
  const result = await spawnFile(command, args, stdin);
  try {
    return JSON.parse(result.stdout) as T;
  } catch (error) {
    const firstLine = result.stdout.trim().split("\n")[0] ?? "";
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`failed to parse mindmemos JSON output: ${message}; stdout=${firstLine}`);
  }
}

function spawnFile(command: string, args: string[], stdin?: string): Promise<SpawnResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["pipe", "pipe", "pipe"],
      // The MindMemOS CLI is Python. On Windows, Python encodes its standard
      // streams with the ANSI code page (e.g. GBK/cp936) when they are not a
      // TTY, so piping here would corrupt non-ASCII output — and read our UTF-8
      // stdin (for `--messages-json-file -`) back as that same code page.
      // Force UTF-8 on all three streams so the `utf-8` decoding below and the
      // UTF-8 stdin round-trip correctly regardless of locale.
      env: {
        ...process.env,
        PYTHONIOENCODING: "utf-8",
        PYTHONUTF8: "1",
      },
    });

    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf-8");
    child.stderr.setEncoding("utf-8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      reject(new Error(`mindmemos exited with code ${code}: ${stderr || stdout}`.trim()));
    });

    if (stdin !== undefined) {
      child.stdin.end(stdin);
    } else {
      child.stdin.end();
    }
  });
}
