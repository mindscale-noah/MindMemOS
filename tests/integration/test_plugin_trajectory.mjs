// Build both plugins, then run with MINDMEMOS_TEST_CLI set to the SDK executable.
import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import openclaw from "../../plugins/openclaw-plugin/dist/index.js";
import { apply } from "../../plugins/deepseek-harness-plugin/dist/index.js";

const cli = process.env.MINDMEMOS_TEST_CLI ?? "mindmemos";
const cwd = process.platform === "win32" ? "C:\\workspace" : "/workspace";
const prompt = "Validate the CSV schema";
const timestamp = 1_790_000_000_000;
const call = (id, name, args) => ({ id, name, args });
const exportCall = (id) => call(id, "bash", { command: `mindmemos skill export ${id} --to skills` });
const showCall = (id) => call(id, "bash", { command: `mindmemos skill show ${id}` });
const readCall = (id, name) => call(id, "read", { file_path: join(cwd, name, "SKILL.md") });

const calls = [
  exportCall("a"), exportCall("b"), exportCall("unused"), showCall("failed"),
  readCall("read-a", "a"), readCall("read-b", "b"), showCall("shown"),
];
const output = (id, content, isError = false) => ({ id, content: JSON.stringify(content), isError });
const results = [
  ...["a", "b", "unused"].map((id) => output(id, { directory: join(cwd, id), skill_version_id: `skv-${id}`, resources: 0 })),
  output("failed", { version: { skill_version_id: "skv-failed", content: "Failed." } }, true),
  output("read-b", "Skill B"),
  output("read-a", "Skill A"),
  output("shown", { version: { skill_version_id: "skv-shown", content: "Skill C" } }),
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
          { role: "assistant", content: calls.slice(4).map(({ id, name, args }) => ({ type: "toolCall", id, name, arguments: args })) },
          ...results.slice(4).map(({ id, content, isError }) => ({ role: "toolResult", toolCallId: id, content, isError })),
        ] : []),
      ],
    }, context),
  };
}

function dshFixture(logger) {
  const hooks = new Map();
  apply({
    skills: { registerProvider() {} },
    logger: () => logger,
    on: hooks.set.bind(hooks),
  }, { cli, sessionId: "override", appId: "dsh", addMode: "async", minQueryLength: 2, topK: 5, maxConversationMessages: 80 });
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
]) {
  const scenario = includeReads ? "several Skill loads" : "Skill exports without a read";
  test(`${host} recalls Memory and reports one turn with ${scenario}`, { timeout: 15_000 }, async (t) => {
    const home = await mkdtemp(join(tmpdir(), "mindmemos-plugin-test-"));
    const previousHome = process.env.MINDMEMOS_HOME;
    process.env.MINDMEMOS_HOME = home;
    const requests = [];
    const server = createServer(async (request, response) => {
      let raw = "";
      for await (const chunk of request) raw += chunk;
      const body = JSON.parse(raw);
      requests.push({ path: request.url, body });
      const [status, receipt] = {
        "/v1/memory/search": [200, { code: "ok", data: { memories: [{ id: "memory-1", memory: "Stored fact" }] } }],
        "/v1/memory/add": [200, { code: "queued", request_id: "memory-1" }],
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
    await writeFile(join(home, "settings.json"), JSON.stringify({
      version: 2,
      service_type: "custom",
      memory_service: { base_url: `http://127.0.0.1:${server.address().port}`, auth: { api_key: "test-key" } },
      defaults: { user_id: "test-user" },
    }));

    const messages = [];
    const receipts = Promise.withResolvers();
    const record = (message) => {
      messages.push(message);
      if (/Skill (stored|failed)/.test(message)) receipts.resolve();
    };
    const plugin = fixture({ info: record, warn: record }, includeReads);
    assert.match(await plugin.recall(), /<relevant-memories>[\s\S]*Stored fact/);
    await plugin.report();
    await receipts.promise;

    assert.deepEqual(requests.map(({ path }) => path), [
      "/v1/memory/search", "/v1/memory/add", "/v1/skills/trajectories/report",
    ]);
    const [search, memory, skill] = requests.map(({ body }) => body);
    assert.equal(search.user_id, undefined);
    assert.equal(memory.user_id, "test-user");
    assert.equal(memory.mode, host === "OpenClaw" ? "sync" : "async");
    assert.equal(memory.session_id, "override");
    assert.equal(memory.messages[0].content, prompt);
    assert.equal(memory.messages[1].tool_calls, undefined);
    assert.equal(skill.event_id, host === "OpenClaw" ? "openclaw:run-7" : "deepseek-harness:host:7");
    assert.equal(skill.agent_id, "agent");
    assert.equal(skill.session_id, "override");
    assert.deepEqual(skill.skill_version_ids, includeReads ? ["skv-b", "skv-a", "skv-shown"] : []);
    assert.equal(skill.messages[1].tool_calls[1].id, "b");
    assert.equal(skill.messages[3].tool_call_id, "b");
    assert.ok(!skill.messages.some(({ content }) => content === "injected"));
    assert.match(messages.join("\n"), /Memory queued.*memory-1/);
    assert.match(messages.join("\n"), skillStatus === 201 ? /Skill stored.*trajectory-1/ : /Skill failed.*skill-rejected/);
  });
}
