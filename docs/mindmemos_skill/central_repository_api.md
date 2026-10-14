# 中心仓 API

`mindmemos_skill` 使用原始正文与独立资源模型。Python 和 HTTP 共用 `SkillService → RepositoryPipeline → SkillPersistence.skills` 调用链，项目身份必须来自可信授权或显式本地绑定。

## Python 本地入口

```python
import asyncio
from mindmemos_skill import create_service
from mindmemos_skill.api.typing import (
    RegisterFromLocalRequest, CommitFromLocalRequest, FileSkillInfo, ExportRequest,
)

async def main():
    async with await create_service("config/mindmemos_skill/dev.yaml") as service:
        first = await service.register_from_local(RegisterFromLocalRequest(
            project_id=service.default_project_id, alias="analysis", skill=FileSkillInfo(path="./my-skill"),
        ))
        second = await service.commit_from_local(CommitFromLocalRequest(
            project_id=service.default_project_id, skill_id=first.skill_id,
            skill=FileSkillInfo(path="./edited-skill", commit_message="Improve instructions"),
        ))
        await service.export_skill(ExportRequest(
            project_id=service.default_project_id, version_id=second.version.version_id, directory="./output/exported",
        ))

if __name__ == "__main__":
    asyncio.run(main())
```

结构化 Python 调用使用 `RegisterRequest`、`CommitRequest` 和 `StructuredSkillInfo`。
本地输入只读取原始 UTF-8 正文和资源，不跟踪后续文件变化；导出不写入仓库元数据。
导出的 `directory` 是父目录，实际写入并返回 `directory/<skill-name>/`。标准格式使用该版本 `SKILL.md` 中的名称，非标准格式使用版本的显式名称；返回的子目录可直接重新注册。
服务工厂返回前自动初始化存储，调用方负责关闭服务。未上线的本地 Skill 数据库仅支持当前结构，清理策略见[持久化说明](../../src/mindmemos_skill/mindmemos_skill/persistence/README.md)。

## 写入边界

- 注册创建新家族和初始版本，同一事务提交；内容相同不会自动覆盖旧家族。
- 提交显式选择父版本，并可选择是否激活；激活指针的并发变化会产生冲突。
- 激活已有版本不创建版本、不写外部目录。家族只保存一个 active 指针。
- 元信息修改不重写版本内容；删除采用软删除，保留已有版本与轨迹。
- 导出默认拒绝非空的同名 Skill 子目录，显式 overwrite 时仅替换该子目录并移除旧文件，保留父目录中的其他内容。发布异常时恢复旧子目录；不提供进程崩溃后的自动恢复。

## 当前内容协议（2026-09-30）

以 `api/entries/fastapi.py` 和运行服务的 `/openapi.json` 为准。

注册：`POST /v1/skills/register`，返回 201。

```json
{
  "alias": "example",
  "skill": {
    "content": "---\nname: Example\ndescription: Example skill\n---\nKeep the original text.\n",
    "resources": {"assets/data.bin": "AP8="},
    "commit_message": "Initial version"
  }
}
```

成功响应的 `data` 包含 `skill_id`、`skill_version_id`、`content_hash`、`skill_hash`。
`skill.standard_format` 默认 true，从 frontmatter 解析名称和描述；false 时须显式传入二者。
`skill.metadata` 保存用户元数据，`skill.commit_version` 指定版本标签，`skill.version_bump` 默认 patch。
传输层拒绝旧 `bundle`、`source` 和 `path`；本地目录仅由 Python `*_from_local` 或本机 UI 读取后转换成结构化内容。

提交：`POST /v1/skills/commit`，在注册的 `skill` 对象外增加 `skill_id`、可选的 `base_version_id`、`active` 和 `inherit_resources`。
`active` 默认 true；资源默认完整替换，显式继承时不允许同时提供非空资源。返回 201，包含新版本 ID、标签、双哈希和 `active_skill_version_id`。

| 路由（均为 POST） | 请求与返回要点 |
| --- | --- |
| `/v1/skills/show` | `skill_id` 和可选 `skill_version_id`；版本返回原始 `content`、直接 Base64 `resources`、名称、描述、格式标记、双哈希和提交元数据。 |
| `/v1/skills/list` | `filters`、可选 `limit`、`cursor`；返回 items、has_more、next_cursor。 |
| `/v1/skills/history` | `skill_id`、可选 `limit`、offset 或 cursor；cursor 绑定项目和家族，固定首次查询时间截止点。 |
| `/v1/skills/activate` | `skill_id`、`skill_version_id`、可选 `expected_active_version_id`；返回 `active_skill_version_id`，不创建版本。 |
| `/v1/skills/version-resolve` | 至少传 `content_hash`、`skill_hash`、`skill_version_id` 之一；可加 `skill_id`，所有条件同时生效。返回双哈希、version、alias、commit_message、is_active。 |
| `/v1/skills/diff` | `skill_id`、`to_skill_version_id`、可选 `from_skill_version_id`、`to_skill_id`；显式检查家族归属。 |

列表、历史和轨迹的 limit 默认 null，表示返回全部匹配项，无固定上界。需要分批读取时显式传正整数。
资源路径/Base64/格式错误与旧请求字段返回 422；不存在或越项目版本返回 404；哈希歧义与过期激活返回 409。
两个相同完整哈希的版本仍有独立版本 ID，不能通过哈希静默合并。

图组织的传输请求统一为 skill_refs、可选 graph_id、alias、commit_message；推荐由 graph_ref、query、top_k 选择。对应算法调用方尚需后续迁移，传输模型可用不代表算法端到端验证完成。
