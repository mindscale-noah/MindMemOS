import React, { useRef, useState } from "react";
import { api } from "./api";
import { useLoad, Message } from "./components";
export function Settings({ dirty }: { dirty: (v: boolean) => void }) {
  const { value, error } = useLoad<any>(() => api("config"), "config");
  const [message, setMessage] = useState("");
  const [failure, setFailure] = useState("");
  const [busy, setBusy] = useState(false);
  const [keySet, setKeySet] = useState<boolean>();
  const form = useRef<HTMLFormElement>(null);
  async function save(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setFailure("");
    setMessage("");
    const data = new FormData(form.current!);
    try {
      const saved = await api<{ api_key_set: boolean }>("config", "PUT", {
        base_url: data.get("base_url"),
        api_key: data.get("api_key") || null,
        clear_api_key: data.get("clear_api_key") === "on",
        user_id: data.get("user_id") || null,
        skill_cache_dir: data.get("skill_cache_dir"),
        skill_backup_dir: data.get("skill_backup_dir"),
        timeout_seconds: Number(data.get("timeout_seconds")),
        max_retries: Number(data.get("max_retries")),
      });
      setKeySet(saved.api_key_set);
      dirty(false);
      (form.current!.elements.namedItem("api_key") as HTMLInputElement).value =
        "";
      (
        form.current!.elements.namedItem("clear_api_key") as HTMLInputElement
      ).checked = false;
      setMessage("配置已保存。");
    } catch (e) {
      setFailure((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function test() {
    setBusy(true);
    setFailure("");
    setMessage("");
    try {
      await api("config/test", "POST");
      setMessage("已保存的配置连接成功。");
    } catch (e) {
      setFailure((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <h1>SDK 配置</h1>
      <p className="muted">
        以下配置用于原 SDK 云端连接，不影响本地 Skill 中心仓。
      </p>
      <Message text={error} error />
      {value && (
        <form ref={form} onSubmit={save} onChange={() => dirty(true)}>
          <fieldset disabled={busy}>
            <section>
              <h2>服务连接</h2>
              <label>
                服务地址
                <input
                  name="base_url"
                  required
                  type="url"
                  defaultValue={value.base_url}
                />
              </label>
              <div className="fields">
                <label>
                  API Key
                  <input
                    name="api_key"
                    type="password"
                    autoComplete="new-password"
                    placeholder={
                      (keySet ?? value.api_key_set)
                        ? "已设置 · 留空保持不变"
                        : "未设置"
                    }
                  />
                </label>
                <label>
                  默认 user_id
                  <input name="user_id" defaultValue={value.user_id || ""} />
                </label>
              </div>
              <label className="check">
                <input type="checkbox" name="clear_api_key" />
                清除已保存的 API Key
              </label>
            </section>
            <section>
              <h2>本地存储</h2>
              <div className="fields">
                <label>
                  缓存目录
                  <input
                    required
                    name="skill_cache_dir"
                    defaultValue={value.skill_cache_dir}
                  />
                </label>
                <label>
                  备份目录
                  <input
                    required
                    name="skill_backup_dir"
                    defaultValue={value.skill_backup_dir}
                  />
                </label>
              </div>
            </section>
            <section>
              <details>
                <summary>网络选项</summary>
                <div className="fields">
                  <label>
                    超时（秒）
                    <input
                      required
                      type="number"
                      min="1"
                      max="600"
                      name="timeout_seconds"
                      defaultValue={value.timeout_seconds}
                    />
                  </label>
                  <label>
                    重试次数
                    <input
                      required
                      type="number"
                      min="0"
                      max="10"
                      name="max_retries"
                      defaultValue={value.max_retries}
                    />
                  </label>
                </div>
              </details>
            </section>
            <div className="row">
              <code className="muted">{value.path}</code>
              <div className="actions">
                <button type="button" onClick={test}>
                  测试已保存配置
                </button>
                <button className="primary">
                  {busy ? "处理中…" : "保存配置"}
                </button>
              </div>
            </div>
          </fieldset>
        </form>
      )}
      <Message text={failure} error />
      <Message text={message} />
    </>
  );
}
