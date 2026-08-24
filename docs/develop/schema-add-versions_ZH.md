# Schema Add 版本切换(v1 / v2)

`algo_config.add.schema.version` 决定 `memory add` 请求走哪条 schema 记忆抽取流程:

| | `v2`(默认) | `v1` |
|---|---|---|
| 流程 | 规则化图融合:每个 episode 一次 entity-generation 调用,按名称/类型确定性判定新建还是更新 | develop 的重 LLM 流程:实体合并决策、高阶属性生成、属性合并/删除、episode 检索字段增强 |
| 每 episode LLM 调用 | 约 5 次(objectify、episode entity、episode edges、schema selection、entity generation) | 更多(增加合并决策、描述更新、高阶、检索字段调用) |
| 数据标签 | `schema_add`(实体 `add_algorithm` / 记忆 `mem_extract_version`) | `schema_add_v1` |
| 分段 prompt | 省 token 版边界 prompt(无 `reasoning` 输出字段) | develop 版边界 prompt(含 `reasoning`) |
| 专属配置 | `merge.description_rewrite_threshold`、`merge.description_max_chars`、`merge.reference_description_max_chars` | `merge.enable_entity_merge_decision`、`use_property_merge`、`secondary_search_*`、`higher_order.*`、`extraction.use_search_fields`、`extraction.episode_search_fields_augment` |

非法取值在配置校验阶段直接拒绝(只允许 `v1` / `v2`)。

## 如何绑定版本

- **部署级默认** — 在基础配置(`config/mindmemos/dev.yaml` 或部署 YAML)里设置
  `algo_config.add.schema.version`,作用于所有没有覆盖值的项目。
- **按项目** — 在该项目 API key 挂载的项目覆盖配置里设置同名字段。网关在每个请求
  解析 key,把 `tenant_config` + `project_config` 合并到基础配置之上(项目优先),
  因此同一部署内不同项目可以并发运行不同版本。Kafka worker 会重新绑定同样的配置
  片段,异步 add 任务与同步路径看到的版本一致。

```yaml
algo_config:
  add:
    schema:
      version: v1   # 默认为 v2
```

## 修改何时生效

- schema-add 运行时(extractor、planner、chunker、prompt)在**每个 drain 循环**从
  请求级配置解析——项目覆盖配置的变更在该项目**下一个 add 请求**即生效,无需重启。
- 修改基础 YAML 需要**重启进程**,基础配置不热加载。
- 版本切换时仍停留在 add buffer 里的记录,按**新**版本切分和抽取;已写入的记忆不会
  因版本切换被改写。

## 存储兼容性

两个版本写同一批 collection,实体/记忆 payload 结构一致,实体召回与版本无关:

- **v2 可以读取并更新 v1 写入的数据,反之亦然。** v2 对 v1 创建的实体做更新时,把
  规则合并(含描述有界增长)应用到既有记录上,无需任何迁移。
- **混合历史与来回切换都是安全的。**
- 差异只在溯源与产物层面,不在结构层面:
  - `mem_extract_version` / `add_algorithm` 不同(`schema_add_v1` vs
    `schema_add`)——两者都是 keyword 索引,可以过滤查询各版本产出的数据,例如
    `mindmemos memory get --filter '{"mem_extract_version":"schema_add_v1"}'`。
  - v1 专属产物(高阶属性、LLM 增强的 episode 检索字段)在任何版本下都保留且可检
    索;v2 只是不再生成新的。

## 生命周期

`v1` 是兼容模式:用于在基准评测中复现 develop 基线,并在 `v2` 验证期间作为回退路径。
至少会保留到 `v2` 在 LoCoMo / PersonaMem 上的结果确认之后;弃用前会提前在 changelog
公告。切换按项目进行、随时可逆——两个方向的存储都是兼容的。

## 回归测试覆盖

- 分段 prompt 一致性 —
  `test_schema_add_pipeline_segments_with_version_matched_boundary_prompt`
- 边界 prompt 版本化 — `test_conv_boundary_detection_prompt_is_versioned`
- 元数据标签一致性 — `test_schema_add_writes_version_matched_labels`
