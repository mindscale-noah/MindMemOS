// Build both plugins, then run with MINDMEMOS_TEST_CLI set to the SDK executable.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { once } from "node:events";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";

import openclaw from "../../plugins/openclaw-plugin/dist/index.js";
import { apply } from "../../plugins/deepseek-harness-plugin/dist/index.js";

const cli = process.env.MINDMEMOS_TEST_CLI ?? "mindmemos";
const cwd = process.platform === "win32" ? "C:\\workspace" : "/workspace";
const prompt = "Validate the CSV schema";
const imageUrl = "https://images.example.test/signed-figure.png";
const timestamp = 1_790_000_000_000;
const call = (id, name, args) => ({ id, name, args });
const exportCall = (id) => call(id, "bash", { command: `mindmemos skill export ${id} --to skills` });
const showCall = (id) => call(id, "bash", { command: `mindmemos skill show ${id}` });
const readCall = (id, name) => call(id, "read", { file_path: join(cwd, name, "SKILL.md") });
const localSource = "---\r\nname: historical-local\r\ndescription: >-\r\n  验证 CSV\r\n---\r\n\r\n旧版正文\r\n";
const nativeSource = "---\nname: native\ndescription: Native Skill\n---\n\nNative instructions.\n";
const digest = (content) => createHash("sha256").update(content, "utf8").digest("hex");
const versions = [
  ["sk-a", "skv-a", "Skill A"],
  ["sk-b", "skv-b", "Skill B"],
  ["sk-shown", "skv-shown", "Skill C"],
  ["sk-local", "skv-local-old", localSource],
  ["sk-local", "skv-local-current", localSource.replace("旧版正文", "新版正文")],
  ["sk-command", "skv-command", "From command."],
  ["sk-id", "skv-id", "From version token."],
  ["sk-wrapped", "skv-wrapped", "Wrapped output."],
  ["sk-hash", "skv-hash", "From returned hash."],
  ["sk-chain", "skv-chain", "From a chained load."],
  ["sk-native", "skv-native", nativeSource],
  ["sk-changed", "skv-changed", "Original export."],
  ["sk-unused", "skv-unused", "Unused."],
].map(([skill_id, skill_version_id, content]) => ({ skill_id, skill_version_id, content_hash: digest(content) }));

const calls = [
  exportCall("a"), exportCall("b"), exportCall("unused"), showCall("failed"),
  readCall("read-a", "a"), readCall("read-b", "b"), showCall("shown"),
];
const output = (id, content, isError = false) => ({ id, content: JSON.stringify(content), isError });
const results = [
  ...["a", "b", "unused"].map((id) => output(id, { directory: join(cwd, id), skill_version_id: `skv-${id}`, resources: 0 })),
  output("failed", { version: { skill_version_id: "skv-failed", content: "Failed." } }, true),
  { id: "read-b", content: "Skill B" },
  { id: "read-a", content: "Skill A" },
  output("shown", { version: { skill_version_id: "skv-shown", content: "Skill C" } }),
];

const additionalCalls = [
  readCall("local", "local"),
  readCall("mismatch", "local"),
  call("command", "exec", { command: 'cd /workspace && "/venv path/mindmemos" skill show sk-command' }),
  call("id", "bash", { command: 'uv run mindmemos skill show "$REF"' }),
  call("wrapped", "bash", { command: "uv run mindmemos skill show wrong-alias" }),
  call("hash", "bash", { command: 'mindmemos skill show "$REF"' }),
  call("chain", "bash", { command: "mindmemos skill show skv-shown && mindmemos skill show skv-chain" }),
  exportCall("changed"),
  readCall("changed-read", "changed"),
  call("resolve-only", "bash", { command: "mindmemos skill resolve skv-unused" }),
  call("commit-only", "bash", { command: "mindmemos skill commit skv-unused --from source" }),
  call("unknown-wrapper", "bash", { command: "wrapper mindmemos skill show skv-unused" }),
  call("show-help", "bash", { command: "mindmemos skill show skv-unused --help" }),
  call("failed-exit", "exec", { command: "mindmemos skill show skv-unused" }),
];
const additionalResults = [
  { id: "local", content: localSource },
  { id: "mismatch", content: localSource.replace("旧版正文", "未注册的正文") },
  { id: "command", content: "Skill source displayed." },
  { id: "id", content: "Loaded version skv-id" },
  { id: "wrapped", content: '[stdout]\n{"version":{"skill_version_id":"skv-wrapped","content":"Wrapped output."}}\n[done]' },
  output("hash", { version: { content_hash: digest("From returned hash.") } }),
  {
    id: "chain",
    content: [
      { version: { skill_version_id: "skv-shown", content: "Skill C" } },
      { version: { skill_version_id: "skv-chain", content: "From a chained load." } },
    ].map((item) => JSON.stringify(item)).join("\n"),
  },
  output("changed", { directory: join(cwd, "changed"), skill_version_id: "skv-changed" }),
  { id: "changed-read", content: "Different content at the exported path." },
  output("resolve-only", { skill_version_id: "skv-unused" }),
  output("commit-only", { version: { skill_version_id: "skv-unused", content: "Unused." } }),
  output("unknown-wrapper", { version: { skill_version_id: "skv-unused", content: "Unused." } }),
  { id: "show-help", content: "usage: mindmemos skill show [-h] skill_ref" },
  { id: "failed-exit", content: "skv-unused\n\n(Command exited with code 1)", isError: true },
];

function openclawFixture(logger, includeReads) {
  const hooks = new Map();
  openclaw.register({ pluginConfig: { cli, sessionId: "override", addMode: "sync" }, logger, on: hooks.set.bind(hooks) });
  const context = { workspaceDir: cwd, runId: "run-7", sessionKey: "host", agentId: "agent" };
  return {
    recall: async () => (await hooks.get("before_prompt_build")({ prompt }, context)).prependContext,
    report: () => hooks.get("agent_end")({
      success: true,
      messages: [
        { role: "user", content: "Prior turn" },
        { role: "user", content: prompt, timestamp },
        { role: "user", content: "<relevant-memories>injected</relevant-memories>" },
        { role: "assistant", content: calls.slice(0, 4).map(({ id, name, args }) => ({ type: "toolCall", id, name, arguments: args })) },
        ...results.slice(0, 4).map(({ id, content, isError }) => ({ role: "toolResult", toolCallId: id, content, isError })),
        ...(includeReads ? [
          { role: "assistant", content: [...calls.slice(4), ...additionalCalls].map(({ id, name, args }) => ({
            type: "toolCall", id, name, arguments: args,
          })) },
          ...[...results.slice(4), ...additionalResults].map(({ id, content, isError }) => ({
            role: "toolResult", toolCallId: id, content, isError: id === "failed-exit" ? undefined : isError,
          })),
        ] : []),
      ],
    }, context),
  };
}

function dshFixture(logger, _includeReads, nativePath) {
  const hooks = new Map();
  const skill = {
    name: "native", description: "Native Skill", content: "Native instructions.",
    path: nativePath, provider: "filesystem", resourceBase: { kind: "directory", path: dirname(nativePath) },
  };
  apply({
    skills: { registerProvider() {}, get: async (name) => name === skill.name ? skill : undefined },
    logger: () => logger,
    on: hooks.set.bind(hooks),
  }, { cli, sessionId: "override", appId: "dsh", addMode: "async", minQueryLength: 2, topK: 5, multimodal: true, maxConversationMessages: 80 });
  const session = { id: "host", header: { cwd, agentPreset: "agent" }, events: [] };
  const event = (type, data) => {
    const item = { type, data, seq: session.events.length, time: timestamp + session.events.length };
    session.events.push(item);
    return item;
  };
  const userMessage = (content, kind = "user") => ({ content: [{ type: "text", text: content }], source: { kind } });
  const toolCalls = (items) => event("assistant/message", { message: { content: items.map(({ id, name, args }) => ({
    type: "tool-call", id, name, arguments: JSON.stringify(args),
  })) } });
  const toolResult = ({ id, content, isError }) => event("tool/result", { message: {
    source: { callId: id },
    content: [{ type: "tool-result", content: [{ type: "text", text: content }], isError }],
  } });
  event("user/message", userMessage("Prior turn"));
  event("turn/start", { turn: 7 });
  event("user/message", userMessage(prompt));
  event("user/message", userMessage("injected", "plugin"));
  toolCalls(calls.slice(0, 4));
  results.slice(0, 4).forEach(toolResult);
  toolCalls(calls.slice(4));
  results.slice(4).forEach(toolResult);
  toolCalls(additionalCalls);
  additionalResults.forEach(toolResult);
  const rendered = [
    '<skill_content name="native">', "<skill_resources>",
    `Base directory for this skill: ${dirname(nativePath)}`,
    "Resolve relative paths mentioned by this skill against the base directory before using them. Load referenced resources only as needed.",
    "</skill_resources>", "", "<skill_instructions>", skill.content, "</skill_instructions>", "</skill_content>",
  ].join("\n");
  event("user/message", {
    content: [{ type: "text", text: rendered }],
    source: { kind: "skill-invocation", name: "native", form: "instructions" },
  });
  const end = event("turn/end", { turn: 7, reason: { kind: "completed" } });
  return {
    recall: async () => {
      const result = await hooks.get("agent/pre-step")({
        agent: session, messages: [userMessage(prompt)], step: 1, signal: new AbortController().signal,
      }, async () => ({ kind: "enter", messages: [] }));
      return result.messages.at(-1).content[0].text;
    },
    report: () => hooks.get("session/event")(session, end),
  };
}

for (const [host, fixture, skillStatus, includeReads] of [
  ["OpenClaw", openclawFixture, 201, true],
  ["DeepSeek Harness", dshFixture, 403, true],
  ["OpenClaw", openclawFixture, 201, false],
  ["OpenClaw", openclawFixture, null, true],
]) {
  const scenario = skillStatus === null
    ? "an unavailable Skill resolver"
    : includeReads ? "several Skill loads" : "Skill exports without a read";
  const expectSkillReport = includeReads && skillStatus !== null;
  test(`${host} recalls Memory and reports one turn with ${scenario}`, { timeout: 15_000 }, async (t) => {
    const home = await mkdtemp(join(tmpdir(), "mindmemos-plugin-test-"));
    const nativePath = join(home, "native", "SKILL.md");
    await mkdir(dirname(nativePath));
    await writeFile(nativePath, nativeSource);
    const previousHome = process.env.MINDMEMOS_HOME;
    process.env.MINDMEMOS_HOME = home;
    const requests = [];
    const server = createServer(async (request, response) => {
      let raw = "";
      for await (const chunk of request) raw += chunk;
      const body = JSON.parse(raw);
      requests.push({ path: request.url, body });
      if (skillStatus === null && request.url.startsWith("/v1/skills/")) {
        response.writeHead(503, { "Content-Type": "application/json" });
        response.end(JSON.stringify({ code: "unavailable" }));
        return;
      }
      const [status, receipt] = {
        "/v1/memory/search": [200, { code: "ok", data: { memories: [
          { id: "memory-1", memory: "Stored fact" },
          { id: "image-1", memory: "", metadata: { image_url: imageUrl } },
        ] } }],
        "/v1/memory/add": [200, { code: "queued", request_id: "memory-1" }],
        "/v1/skills/skill-resolve": (() => {
          const reference = body.skill_ref === "native" ? "sk-native" : body.skill_ref;
          const selected = versions.find((version) => version.skill_id === reference || version.skill_version_id === reference);
          return selected ? [200, { code: "ok", data: selected }] : [404, { code: "not_found" }];
        })(),
        "/v1/skills/list": [200, { code: "ok", data: {
          items: body.filters?.name === "historical-local" ? [{ name: "historical-local", skill_id: "sk-local" }] : [],
        } }],
        "/v1/skills/version-resolve": (() => {
          const matches = versions.filter((version) => ["skill_id", "skill_version_id", "content_hash"].every(
            (key) => !body[key] || version[key] === body[key],
          ));
          return matches.length === 1 ? [200, { code: "ok", data: matches[0] }] : [404, { code: "not_found" }];
        })(),
        "/v1/skills/trajectories/report": skillStatus === 201
          ? [201, { code: "ok", data: { trajectory_id: "trajectory-1", skill_version_ids: body.skill_version_ids } }]
          : [403, { code: "permission_denied", request_id: "skill-rejected" }],
      }[request.url] ?? [404, {}];
      response.writeHead(status, { "Content-Type": "application/json" });
      response.end(JSON.stringify(receipt));
    });
    t.after(async () => {
      server.close();
      if (previousHome === undefined) delete process.env.MINDMEMOS_HOME;
      else process.env.MINDMEMOS_HOME = previousHome;
      await rm(home, { recursive: true, force: true });
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const baseUrl = `http://127.0.0.1:${server.address().port}`;
    await writeFile(join(home, "settings.v2.json"), JSON.stringify({
      version: 2,
      service_type: "custom",
      memory_service: { base_url: baseUrl, auth: { api_key: "test-key" } },
      skill_service: { base_url: `${baseUrl}/v1/skills`, auth: { api_key: "test-key" } },
      defaults: { user_id: "test-user" },
    }));

    const messages = [];
    const receipts = Promise.withResolvers();
    const record = (message) => {
      messages.push(message);
      if (/Skill (stored|skipped|failed)/.test(message)) receipts.resolve();
    };
    const plugin = fixture({ info: record, warn: record }, includeReads, nativePath);
    const context = await plugin.recall();
    assert.match(context, /<relevant-memories>[\s\S]*Stored fact/);
    assert.ok(context.includes(`<image: ${imageUrl}>`));
    await plugin.report();
    await receipts.promise;

    const deliveries = requests.filter(({ path }) => path.startsWith("/v1/memory/") || path.endsWith("/trajectories/report"));
    assert.deepEqual(deliveries.map(({ path }) => path), [
      "/v1/memory/search", "/v1/memory/add", ...(expectSkillReport ? ["/v1/skills/trajectories/report"] : []),
    ], messages.join("\n"));
    const [search, memory, skill] = deliveries.map(({ body }) => body);
    assert.equal(search.user_id, undefined);
    assert.equal(search.multimodal, true);
    assert.equal(memory.user_id, "test-user");
    assert.equal(memory.mode, host === "OpenClaw" ? "sync" : "async");
    assert.equal(memory.session_id, "override");
    assert.equal(memory.messages[0].content, prompt);
    assert.equal(memory.messages[1].tool_calls, undefined);
    assert.match(messages.join("\n"), /Memory queued.*memory-1/);
    if (!expectSkillReport) {
      assert.match(messages.join("\n"), /Skill skipped.*no_skill_versions/);
      return;
    }
    assert.equal(skill.event_id, host === "OpenClaw" ? "openclaw:run-7" : "deepseek-harness:host:7");
    assert.equal(skill.agent_id, "agent");
    assert.equal(skill.session_id, "override");
    assert.deepEqual(skill.skill_version_ids, [
      "skv-b", "skv-a", "skv-shown", "skv-local-old", "skv-id", "skv-wrapped", "skv-hash", "skv-chain",
      ...(host === "DeepSeek Harness" ? ["skv-native"] : []),
    ]);
    assert.ok(requests.some(({ path, body }) => path.endsWith("/version-resolve")
      && body.skill_id === "sk-local" && body.content_hash === digest(localSource)));
    assert.equal(skill.messages[1].tool_calls[1].id, "b");
    assert.equal(skill.messages[3].tool_call_id, "b");
    assert.ok(!skill.messages.some(({ content }) => content === "injected" || content.startsWith("<skill_content")));
    assert.match(messages.join("\n"), skillStatus === 201 ? /Skill stored.*trajectory-1/ : /Skill failed.*skill-rejected/);
  });
}
