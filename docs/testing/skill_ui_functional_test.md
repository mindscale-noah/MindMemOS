# Skill 与 UI 功能测试

> 当前先提供一个完整场景，用于确认转测文档的书写方式。后续场景沿用“服务前置、操作流程、预期结果”的结构。

## 场景 01：通过 UI 完成 Skill 注册、编辑、版本切换、导出和删除

以“会议纪要”Skill 为主线完成整个管理流程，同时注册“费用审核”和“发布检查”Skill，检查多条记录的展示、搜索以及操作隔离。

### ① 服务前置

**服务准备**

- 使用本地测试环境，已按[部署指南](../deploy/instruction_ZH.md)安装依赖、初始化本地配置并构建 UI，Docker 已启动。
- 若测试环境尚未运行，在仓库根目录执行下面的命令并保持终端运行；已有可用测试服务时直接使用。

```bash
make dev SERVICE=skill FEATURE=core
```

- 打开 <http://127.0.0.1:8765>，确认左下角 `Skill Available`。本场景不需要 Memory 服务或模型调用。
- 进入 `Skills`，确认当前测试项目没有别名为 `qa-meeting-notes`、`qa-expense-review`、`qa-release-check`、`qa-meeting-notes-copy` 的记录。若存在上轮测试记录，先通过详情页的 `More` → `Delete Skill` → `Confirm deletion` 删除这些测试记录，再开始本轮；无需清空数据库。

**生成测试文件**

新开终端，在仓库根目录执行一次：

```bash
uv run --no-sync python .skill_test/prepare.py
```

脚本在仓库 `.skill_test/` 下生成以下文件，并打印三份 Skill 和导出目标的**绝对路径**，后续直接复制到页面输入框。脚本只生成本地文件，不会向服务注册 Skill。

| 测试 Skill | 本地目录 | 辅助资源 | 用途 |
| --- | --- | --- | --- |
| 会议纪要 | `.skill_test/skills/qa-meeting-notes/` | `resources/meeting.md`、`resources/template.md`、`references/checklist.md` | 主流程：编辑、版本管理、导出 |
| 费用审核 | `.skill_test/skills/qa-expense-review/` | `resources/expenses.csv`、`references/policy.md` | CSV 预览、列表搜索、操作隔离 |
| 发布检查 | `.skill_test/skills/qa-release-check/` | `resources/release.json`、`references/checklist.md` | JSON 预览、列表搜索、操作隔离 |

每份 Skill 均有 `SKILL.md`。脚本还准备了导出目录 `.skill_test/exports/qa-meeting-notes/`，其中放置 `obsolete.md`，用于检查覆盖导出是否移除旧文件。

脚本和三份 Skill 源文件均可纳入 Git；`exports/` 是本地测试产物，已忽略。重复执行脚本只补齐缺失文件，不覆盖已有文件。下文编辑均在 UI 中完成，本地源文件应保持初始内容；若曾手动改过源文件，先恢复初始测试文件再执行。

### ② 操作流程与③预期结果

#### 01-01 注册三份 Skill，检查列表和文件内容

**操作流程**

1. 在 `Skills` 页面点击 `＋ Register Skill`。
2. `Skill folder` 粘贴脚本输出的 `qa-meeting-notes` 绝对路径。
3. `Alias (optional)` 填写 `qa-meeting-notes`，`Initial version` 填写 `v1.0.0`，`Commit message (optional)` 填写 `首次注册会议纪要 Skill`。
4. 点击 `Preview registration`，核对名称、目录、服务地址，再点击 `Confirm registration`。
5. 按相同步骤注册另外两份 Skill，别名分别填写 `qa-expense-review`、`qa-release-check`，初始版本均填写 `v1.0.0`。
6. 依次进入三份 Skill 的 `Content`，点击文件树中的正文和每个辅助文件，检查内容。

**预期结果**

- 列表新增三条记录，名称和别名对应；每条的 `Active version`、`Latest version` 均为 `v1.0.0`。
- 会议纪要 Skill 共四个文件；另外两份各三个文件，目录层级与前置表格一致。
- 会议记录中能看到 Alice、Bob 的任务；CSV 中能看到 80、120、200 三条金额；JSON 中能看到 `smoke_test_passed: false`。
- 三份 Skill 的文件内容互不混淆，页面没有读取错误。

#### 01-02 搜索 Skill，并检查重复注册

**操作流程**

1. 返回列表，在搜索框输入 `qa-expense-review`，检查结果，再清空搜索框。
2. 再次点击 `＋ Register Skill`，选择原会议纪要目录，别名仍填写 `qa-meeting-notes`，初始版本仍填写 `v1.0.0`。
3. 点击 `Preview registration`；如果预览成功，继续点击 `Confirm registration`，检查最终提示。
4. 关闭窗口，点击列表上方的 `Refresh`，再进入原会议纪要 Skill 的 `Version history`。

**预期结果**

- 搜索后只显示费用审核 Skill；清空后恢复本页记录。当前搜索范围为当前页。
- 重复注册被拒绝，页面提示别名冲突，不新增重复记录。
- 原会议纪要 Skill 仍只有一个 `v1.0.0`，正文与资源不变。

#### 01-03 编辑正文和辅助文件，提交但不启用

**操作流程**

1. 进入会议纪要 Skill 的 `Content`，确认 `Version` 为 `v1.0.0`，点击 `Edit`。
2. 选择 `SKILL.md`，在正文末尾增加：`4. Include a deadline for each action item.`，保留顶部名称与描述。
3. 选择 `references/checklist.md`，在末尾增加：`- Include a deadline for each action item.`。
4. 点击 `Review changes`，检查这两个文件的新增内容，再点击 `Back to editing`。
5. `Version numbering` 保持 `Patch +1`，`Commit message` 填写 `补充待办截止日期`，点击 **`Commit`**。
6. 提交完成后返回列表；重新进入详情，在 `Version` 中分别选择 `v1.0.0`、`v1.0.1`，核对正文和检查清单。

**预期结果**

- 新增 `v1.0.1`，提交说明为“补充待办截止日期”。
- 列表的当前启用版本仍为 `v1.0.0`，最新版本为 `v1.0.1`。
- `v1.0.1` 的两个文件均包含新增要求；`v1.0.0` 保持原内容。
- `resources/meeting.md`、`resources/template.md` 保留且内容不变。

#### 01-04 比较版本，并启用新版本

**操作流程**

1. 点击 `Compare`，将 `Base version` 设为 `v1.0.0`，`Target version` 设为 `v1.0.1`。
2. 检查 `SKILL.md` 和 `references/checklist.md` 的差异。
3. 返回 `Content`，选择 `v1.0.1`，点击 `Activate v1.0.1` → `Confirm activation`，成功后关闭窗口。
4. 返回列表，点击 `Refresh`。

**预期结果**

- 差异中准确显示两个文件新增的截止日期要求，其他文件没有变化。
- 当前启用版本和最新版本均为 `v1.0.1`。
- `Version history` 仍只有两个版本；比较和启用操作不新增版本。

#### 01-05 更新描述，新增和删除资源文件，提交并启用

**操作流程**

1. 在 `Content` 选择 `v1.0.1`，点击 `Edit`。
2. 打开 `SKILL.md`，将顶部 `description` 改为 `Summarize meetings with owners and deadlines.`，保留 `name`。
3. 点击文件树上方的 `＋` → `New file`，在 `Relative path` 填写 `resources/deadlines.md`，点击 `Create file`。在新文件中输入 `Every action item must include an owner and a deadline.`。
4. 点击 `resources/template.md` 旁的删除按钮，在弹窗确认删除。同步将 `SKILL.md` 第 2 条改为 `2. Write a summary with owners and deadlines.`，避免继续引用已删除的模板。
5. 点击 `Review changes`，核对正文修改、新文件和模板删除，再返回编辑。
6. `Version numbering` 保持 `Patch +1`，提交说明填写 `更新描述和辅助资源`，点击 **`Commit & activate`**。
7. 查看 `v1.0.2` 的 `File metadata` 和文件树，再切换到 `v1.0.1` 对照。

**预期结果**

- 新增并启用 `v1.0.2`；当前启用版本和最新版本均为 `v1.0.2`。
- `File metadata` 中显示新的 `description`，名称保持 `qa-meeting-notes`。
- `v1.0.2` 包含 `resources/deadlines.md`，不再包含 `resources/template.md`，共四个文件。
- 旧版本 `v1.0.1` 仍有原描述和模板，不含新文件；会议原文和检查清单未丢失。

这里更新的是 `SKILL.md` 中的文件描述。家族级别名和 metadata 当前没有 UI 编辑入口，另在后续 CLI 场景覆盖。

#### 01-06 切回历史版本，再恢复最新版本

**操作流程**

1. 在 `Content` 选择 `v1.0.0`，点击 `Activate v1.0.0` → `Confirm activation`。
2. 关闭窗口，返回列表刷新，再进入详情查看正文和资源。
3. 选择 `v1.0.2`，按相同步骤重新启用。

**预期结果**

- 切回后，当前启用版本为 `v1.0.0`，最新版本仍为 `v1.0.2`；初始内容可完整查看。
- 恢复后，当前启用版本为 `v1.0.2`，新增资源和新描述均存在。
- 历史始终只有三个版本，另外两份 Skill 仍停留在 `v1.0.0`。

#### 01-07 导出到本地，检查完整资源和覆盖行为

**操作流程**

1. 在 `Content` 选择 `v1.0.2`，点击 `Export v1.0.2`。
2. `Destination Skill folder` 粘贴脚本输出的 `Export destination` 绝对路径。
3. 点击 `Preview export`，核对路径以及替换目录的提示，再点击 `Confirm export`。
4. 在系统文件管理器中打开该路径，检查导出的正文、资源文件和旧文件 `obsolete.md`。

**预期结果**

- 预览提示目标目录将被替换，确认后显示 `Exported`。
- 导出目录包含当前版本的四个文件，新描述、截止日期要求、新资源内容均与页面一致。
- `obsolete.md` 已被移除，已删除的模板也不在导出目录中。
- `.skill_test/skills/qa-meeting-notes/` 原始目录仍保留初始文件，未被 UI 编辑或导出修改。

#### 01-08 从导出目录重新注册，检查副本独立性

**操作流程**

1. 返回列表，点击 `＋ Register Skill`，`Skill folder` 填写刚才的导出目录。
2. 别名填写 `qa-meeting-notes-copy`，初始版本填写 `v1.0.0`。
3. 预览确认名称仍为 `qa-meeting-notes`，点击 `Confirm registration`。
4. 按 Alias 区分原 Skill 和副本，分别查看内容与版本历史。

**预期结果**

- 列表新增一条独立副本，名称相同但别名不同。
- 副本只有一个 `v1.0.0`，其正文与资源等同于原 Skill 的 `v1.0.2`。
- 原 Skill 仍有三个版本；费用审核、发布检查两份 Skill 不受影响。

#### 01-09 取消删除、确认删除，并清理本轮数据

**操作流程**

1. 进入别名为 `qa-meeting-notes-copy` 的副本，点击 `More` → `Delete Skill`，先点击 `Cancel`。
2. 返回列表确认副本仍在，再进入副本详情重复删除操作，点击 `Confirm deletion`。
3. 返回列表并清空搜索条件，点击 `Refresh`；打开其他三份 Skill，确认仍可查看。
4. 通过同样的页面操作删除本轮注册的其余三份测试 Skill。
5. 在文件管理器中检查 `.skill_test/skills/` 和 `.skill_test/exports/qa-meeting-notes/`。

**预期结果**

- 取消删除不会改变记录；确认删除后副本从列表消失。
- 删除副本不影响其他三份 Skill 的内容和版本。
- 清理后列表中不再出现本轮四个测试别名，其他项目原有记录不受影响。
- 本地源文件与导出文件仍存在。

### 测试记录

| 步骤 | 检查内容 | 结果（通过/失败/阻塞） | 实际结果或缺陷编号 |
| --- | --- | --- | --- |
| 01-01 | 注册、列表、Markdown/CSV/JSON 资源查看 | | |
| 01-02 | 搜索和重复注册拒绝 | | |
| 01-03 | 编辑正文与资源、提交但不启用 | | |
| 01-04 | 差异比较和启用版本 | | |
| 01-05 | 描述更新、资源增删、提交并启用 | | |
| 01-06 | 历史版本切换和恢复 | | |
| 01-07 | 完整导出、覆盖旧文件 | | |
| 01-08 | 导出副本重新注册 | | |
| 01-09 | 取消删除、删除隔离和清理 | | |

全部步骤符合预期后，本场景通过。前置服务不可用记为“阻塞”；操作结果与预期不符记为“失败”，记录实际表现和页面截图。
