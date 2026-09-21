// Run after npm run build with Playwright available on NODE_PATH.
// Optional: PLAYWRIGHT_CHROMIUM_EXECUTABLE selects an existing Chromium install.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

test("navigation history, retained filters, direct links and unsaved settings", async () => {
  const root = path.resolve(__dirname, "../../mindmemos_ui/static");
  const server = http.createServer((req, res) => {
    const file = path.join(
      root,
      req.url === "/" ? "index.html" : req.url.split("?")[0],
    );
    res.setHeader(
      "Content-Type",
      file.endsWith(".js")
        ? "text/javascript"
        : file.endsWith(".css")
          ? "text/css"
          : "text/html",
    );
    res.end(fs.readFileSync(file));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
    });
    const page = await browser.newPage();
    page.setDefaultTimeout(8000);
    page.setDefaultNavigationTimeout(8000);
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const skill = {
      skill_id: "skill / 中文",
      name: "Navigation skill",
      current_version_id: "v2",
      current_version_label: "v0.0.2",
      latest_version_id: "v2",
      latest_version_label: "v0.0.2",
    };
    const versions = ["v1", "v2"].map((id, i) => ({
      version_id: id,
      version_label: `v0.0.${i + 1}`,
      created_at: "2026-09-21",
      origin: "test",
      status: "ready",
    }));
    await page.route("**/api/**", async (route) => {
      const url = new URL(route.request().url());
      const key = decodeURIComponent(url.pathname.slice(5));
      let data;
      if (key === "session")
        data = { token: "test", home: "/test", graphs_connected: true };
      else if (key === "skills") data = [skill];
      else if (key.startsWith("skills/"))
        data = { skill, version: versions[1] };
      else if (key === "versions") data = versions;
      else if (key === "implementation")
        data = { implementation: "static", dynamic: false };
      else if (key === "content")
        data = { files: { "SKILL.md": "# Example" }, resources: [] };
      else if (key === "graphs")
        data = [{ graph_id: "g1", name: "Test graph" }];
      else if (key === "graphs/g1/snapshots")
        data = [
          { graph_version_id: "s1", revisions: [1], created_at: "2026-09-20" },
          { graph_version_id: "s2", revisions: [2], created_at: "2026-09-21" },
        ];
      else if (key.startsWith("graphs/g1/snapshots/"))
        data = {
          nodes: [
            {
              node_id: "n1",
              skill_id: skill.skill_id,
              skill_version_id: "v1",
              name: skill.name,
            },
          ],
          edges: [],
        };
      else if (key === "configurations/default-home") data = { home: "/test" };
      else if (key === "client-configurations")
        data = {
          home: "/test",
          runtime_home: "/test",
          memory: [
            { name: "default", exists: true, path: "/test/settings.json" },
          ],
          skill: [],
        };
      else if (key === "client-configurations/memory/default")
        data = {
          name: "default",
          kind: "memory",
          exists: true,
          path: "/test/settings.json",
          env_path: "/test/settings.env",
          revision: "test",
          env: [],
          content: "user_id: original\n",
          values: { base_url: "http://127.0.0.1:8000", api_key: null, user_id: "original", timeout_seconds: 21600 },
        };
      else if (key === "config")
        data = {
          base_url: "https://example.com",
          skill_cache_dir: "/cache",
          skill_backup_dir: "/backup",
          timeout_seconds: 30,
          max_retries: 2,
        };
      else throw new Error(`Unexpected API: ${key}`);
      await route.fulfill({ json: data });
    });
    const base = `http://127.0.0.1:${server.address().port}/`;
    const waitHash = (text) =>
      page.waitForURL((url) => url.hash.includes(text));
    await page.goto(base);
    await page.getByRole("textbox", { name: "搜索 Skill" }).fill("Navigation");
    await page.getByRole("button", { name: skill.name, exact: true }).click();
    await waitHash("skill=");
    const detailUrl = page.url();
    await page.goBack();
    await page.getByRole("heading", { name: "Skills 中心仓" }).waitFor();
    assert.equal(
      await page.getByRole("textbox", { name: "搜索 Skill" }).inputValue(),
      "Navigation",
    );
    await page.goForward();
    await page.getByRole("button", { name: "版本历史", exact: true }).click();
    await waitHash("tab=versions");
    await page.getByRole("button", { name: "v0.0.1", exact: true }).click();
    await waitHash("version=v1");
    await page.goBack();
    await waitHash("tab=versions");
    await page.goForward();
    await waitHash("version=v1");
    await page.getByRole("button", { name: "← 返回", exact: true }).click();
    await page.getByRole("heading", { name: "Skills 中心仓" }).waitFor();
    assert.equal(
      await page.getByRole("textbox", { name: "搜索 Skill" }).inputValue(),
      "Navigation",
    );

    await page.getByRole("button", { name: "管理图谱", exact: true }).click();
    await page.getByLabel(/图谱快照/).selectOption("s1");
    await page.getByRole("textbox", { name: "定位 Skill" }).fill("Navigation");
    await page.locator(".graph-node").click();
    await page.getByRole("button", { name: "打开此版本" }).click();
    await page.getByRole("button", { name: "← 返回", exact: true }).click();
    await page.getByLabel(/图谱快照/).waitFor();
    assert.equal(await page.getByLabel(/图谱快照/).inputValue(), "s1");
    assert.equal(
      await page.getByRole("textbox", { name: "定位 Skill" }).inputValue(),
      "Navigation",
    );
    await page.getByRole("button", { name: "打开此版本" }).waitFor();

    await page.getByRole("button", { name: "配置", exact: true }).click();
    const openSettings = async () => {
      if (await page.getByText("请先选择配置目录", { exact: true }).count()) {
        await page.getByRole("button", { name: "手动输入路径", exact: true }).click();
        await page.getByLabel("HOME 路径", { exact: true }).fill("/tmp/navigation-home");
        await page.getByRole("button", { name: "使用此路径", exact: true }).click();
      }
      await page.getByRole("button", { name: /M Memory 配置/ }).click();
    };
    await openSettings();
    await page.getByRole("textbox", { name: /^默认用户/ }).fill("unsaved");
    page.once("dialog", (dialog) => dialog.dismiss());
    await page.goBack();
    await waitHash("page=settings");
    assert.equal(
      await page.getByRole("textbox", { name: /^默认用户/ }).inputValue(),
      "unsaved",
    );
    page.once("dialog", (dialog) => dialog.dismiss());
    await page.evaluate(() => history.go(-2));
    await waitHash("page=settings");
    assert.equal(
      await page.getByRole("textbox", { name: /^默认用户/ }).inputValue(),
      "unsaved",
    );
    page.once("dialog", (dialog) => dialog.dismiss());
    await page.getByRole("textbox", { name: /^默认用户/ }).press("Escape");
    assert.equal(await page.getByRole("textbox", { name: /^默认用户/ }).isVisible(), true);
    page.once("dialog", (dialog) => dialog.accept());
    await page.goBack();
    await waitHash("page=graphs");
    await page.goForward();
    await openSettings();
    assert.equal(
      await page.getByRole("textbox", { name: /^默认用户/ }).inputValue(),
      "original",
    );
    await page.getByRole("button", { name: "关闭", exact: true }).click();
    await page.getByRole("button", { name: "Skills", exact: true }).click();
    await page.goBack();
    await openSettings();
    await page.getByRole("textbox", { name: /^默认用户/ }).fill("forward-draft");
    page.once("dialog", (dialog) => dialog.dismiss());
    await page.goForward();
    await waitHash("page=settings");
    assert.equal(
      await page.getByRole("textbox", { name: /^默认用户/ }).inputValue(),
      "forward-draft",
    );
    page.once("dialog", (dialog) => dialog.accept());
    await page.goForward();
    await page.getByRole("heading", { name: "Skills 中心仓" }).waitFor();

    await page.goto(detailUrl);
    await page
      .getByRole("button", { name: "提交新版本", exact: true })
      .waitFor();
    await page.reload();
    await page
      .getByRole("button", { name: "提交新版本", exact: true })
      .waitFor();
    await page.getByRole("button", { name: "← 返回", exact: true }).click();
    await page.getByRole("heading", { name: "Skills 中心仓" }).waitFor();
    if (process.env.NAVIGATION_SCREENSHOT)
      await page.screenshot({
        path: process.env.NAVIGATION_SCREENSHOT,
        fullPage: true,
      });
    assert.deepEqual(errors, []);
  } finally {
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
  }
});
