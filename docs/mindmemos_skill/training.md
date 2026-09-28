# 通用 Skill 训练流程

`training` 是一个通用 TrainingPipeline 实现，配置中的 `trainer.class_name` 可选
`treeskill` 或 `skill_forest`，也可选择显式注册的其他 `SkillTrainer`。Pipeline 不按算法名
分支，不读取算法检查点，也不传递 `resume` 参数。

完整服务示例见 `config/mindmemos_skill/training/service.yaml`。其中两个命名入口都使用
`class_name: training`，分别配置 Trainer 和 Environment。模型名称为占位值，运行前替换。
当前通过本地 Python 服务入口训练；FastAPI 不开放训练路由，也没有 UI 训练表单或后台任务队列。

## 调用与数据划分

```python
from mindmemos_skill import create_service
from mindmemos_skill.pipelines.typing import TrainingRequest, TrainingSplit

async with await create_service("config/mindmemos_skill/training/service.yaml") as service:
    result = await service.auto_train_skill(
        TrainingRequest(
            pipeline="skill_forest",  # Change to "treeskill" to use the same business flow.
            project_id="my-project",
            base_version_id=existing_version_id,
            work_dir="output/training/run-001",
            tasks=materialized_samples,
            split=TrainingSplit(train=train_ids, val=val_ids, test=test_ids),
        )
    )
```

`tasks` 使用公共 `DataSample`，包含公开任务和独立私有评分输入。所有任务必须且只能分配到
train/val/test 的一个集合。请求必须提供工作目录；相对路径以服务进程工作目录为基准。
Python 调用会等待训练及发布完成。调用方须准备完整 `DataSample` 与 train/val/test 划分；私有评分输入不通过 HTTP 请求传输。

Pipeline 校验输入版本的项目归属和当前指针，读取完整快照，将 train 与 val 交给 Trainer，
始终将 test 留在 Trainer 外。最终结果冻结后，Pipeline 对非空 val/test 各评估一次并记录指标，
不根据它们的分数再次选优、修改结果或决定版本是否发布。执行异常阻止发布，不伪装为零分。
TreeSkill 当前不使用外部 val 优化；Forest 在 train 池内部为每棵树划分训练和验证，外部 val
也不参与森林选优。切换其他 Trainer 时，其内部 val 语义由该算法定义。

Forest 当前只接受静态源 Skill，避免把动态实现元数据当作已恢复的运行时行为。
TreeSkill 从源正文训练并返回带路由配置的 TreeSkill，最终评估使用该具体对象；快照入库后保留
其 implementation、配置和 resources。评估使用独立工作区和 `evaluation_concurrency` 限制。

## 自动恢复归属 Trainer

再次调用同一算法、相同输入和相同 `work_dir` 即自动复用兼容工作，没有恢复开关。
Pipeline 始终把 `<work_dir>/train` 传给 Trainer；是否恢复、恢复哪些步骤由 Trainer 自行决定。

- 空目录或不存在的目录：新建训练。
- 兼容且未完成：复用已提交的调用结果，继续未完成工作。
- 兼容且已完成：返回已保存的最终 Skill，TreeSkill 恢复具体动态类及路由依赖。
- 输入、算法、配置、源码指纹变化，或检查点损坏：报错并保留目录。
- 只有旧审计文件、缺少恢复协议的历史目录：报错，需使用新目录，不能直接覆盖。

TreeSkill 复用已提交的采样成功/失败结果和模型响应，重新执行确定性的分析/融合编排；
终态失败不会因恢复获得额外随机采样机会。未提交完成记录的物理请求可能重跑，旧 attempt
保留。Forest 还复用候选生成、验证、波次与集成边界。两者均不恢复环境内部的半条交互轨迹，
也不保证远端模型计费请求 exactly-once。运行目录锁阻止同时写入。

`request.json` 绑定业务输入和配置，`selected_skill.json` 保存发布前的完整候选；
`val/`、`test/` 是独立评估产物，`publication.json` 是成功发布的结果。数据库提交失败时
候选保留；再次调用时 Trainer 与评估函数各自复用兼容结果。

## 发布与既有数据库兼容

发布使用启动时捕获的父版本，训练期间不持有数据库事务；提交时在短事务内核验当前版本。
其他写入推进了当前版本时，发布失败并保留候选，不自动覆盖。版本标签由持久层分配，算法
后缀为 `treeskill` 或 `skill-forest`，注册名中的下划线统一转换成标签允许的连字符。

同一项目与工作目录产生稳定 run_id。发布版本使用确定性内部 ID，并在已有
`local_metadata.training` 字段记录 run_id 和输入指纹；版本插入、标签分配与当前指针更新
处于同一事务。即使提交成功后进程中断、调用方未收到结果，重试也返回原版本，不新增版本。
如果之后用户已切换或提交其他版本，重试旧请求不改变当前指针。相同 run_id 配不同输入明确冲突。

此功能不新增表或列，不修改历史记录、不更改商用云服务数据契约；无需数据回填或数据库重建。
部署继续使用原数据库，按既有方式初始化缺失表和版本标签；旧代码可以继续读取新版本。
回退服务代码不需要回退数据。只需将原本未启用的训练流程配置为 `training`，并在训练请求中
增加 `work_dir`；未配置训练的服务行为保持不变。
