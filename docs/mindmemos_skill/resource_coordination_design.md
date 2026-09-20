# 第一版并发与恢复边界

状态：以本文件取代此前通用资源锁实施方案。第一版不实现 resource_locks、coordination 服务、长任务 lease、续租、超时接管或 fencing token。

## 1. 第一版保留的能力

| 机制 | 用途 | 实施边界 |
| --- | --- | --- |
| 不可变版本 | 固定输入，独立生成候选 | 每次尝试新建 graph_version_id 或 Skill 候选版本 |
| 短事务发布与 expected_revision | 防止陈旧结果覆盖新发布 | 发布事务锁图的当前记录，校验后原子更新 |
| 成功请求幂等 | 重试返回原结果 | revisions 保存 idempotency_key 与 request_hash；同图唯一 |

默认流程：读取固定输入 → 独立计算 → 写入独立候选并封存 → 短事务发布。允许重复计算，不允许同一请求重复成功发布。

## 2. Graph

同一个候选只有创建它的执行流程写入，不向用户开放任意候选写 API。该流程完成全部批次后才封存；重试使用新候选，不接管原候选。第一版不做候选自动 GC，因此不引入候选清理与运行中写入的自动竞争机制。

最终发布在同一事务内锁 graph 行、重新检查幂等结果、验证 expected_revision、检查 ready，插入 revision 并更新 current_revision。新 key 的无内容变化请求仍新增 revision，复用图内容；同 key 重放不新增 revision。详见 [图存储结构](graph_storage_schema.md)。

这些是待实现服务契约。当前阶段先实现五张图表、字段验证、约束和表注册，不把通用数据库 CRUD 描述为已经具备完整发布协议。

## 3. 单 Skill 演进

trace2skill、skilltree、skill-forest 均按“固定 Skill + 固定轨迹/任务批次 → 单一候选 Skill”处理。计算阶段不锁 Skill；树和森林属于算法内部搜索状态。

作为库被宿主调用时，返回候选，由宿主决定是否采用并负责条件发布。本服务以后需要管理 Skill 当前版本时，再定义对应发布指针与幂等结果，不把 skill_versions.version_revision 误用为 Skill family 的发布序号。

## 4. 恢复

文件 checkpoint 按算法的实际需求实现，保存固定输入及摘要、算法版本与配置、候选与评估状态、迭代位置、随机状态和预算使用。独立执行尝试使用独立目录；恢复从完整 checkpoint 读取，在新目录与新候选上继续，不让两个进程覆盖同一工作文件。

采用完整临时文件写入后原子替换；多文件状态最后更新 manifest。跨实例恢复要求文件可被接手实例访问，本地临时目录不能保证实例销毁后的恢复。恢复算法进度不表示恢复发布权，最后仍检查 expected_revision 与成功幂等结果。

## 5. 暂缓的机制

- 通用锁表、长 lease、heartbeat、fencing 与后台接管。
- 相同输入任务合并、持久化任务队列及失败结果查询。
- 项目预算预留、分布式并发额度与自动候选 GC。

以后确有跨算法资源协调需求再提取公共机制，不提前将其作为所有算法的必经依赖。开放付费公网接口前再补入口限流与请求规模限制；成功幂等只能防重复发布，不能防恶意换 key 发起计算。

## 6. 模块边界

persistence/db_services/skill_management/ 定义五张图表及其记录类型，后续同目录实现图业务查询和发布。顶层 persistence 负责工厂、ensure_collection 与注册；infra 只提供连接、事务和存储类型，不放业务表。

本轮不创建 coordination 模块，不修改商用 mindmemos 云表，不执行已有数据库迁移。ensure_collection 创建缺失表，不能代替 schema migration。
