# Customer Support seed42 自演化记录与效果分析

本文整理 `online-controlled` 实验中 Customer Support、seed 42 的十次自演化，回答三个问题：每轮收到了什么反馈、配置具体改变了什么、规划器为什么做出这些改变。原始实验产物位于
`outputs/statebench/customer_support/seed42/`。

## 实验口径

- Baseline 与 Evolution 都在线运行相同的 100 个官方训练任务，并正常执行记忆检索和添加。
- Baseline 固定使用初始配置，不执行 feedback collect 和 self-evolve。
- Evolution 每完成 10 个训练任务，统一收集这 10 条轨迹中的反馈，再执行一次 self-evolve；新配置从下一轮训练开始生效。
- 测试阶段只检索，不添加记忆、不收集反馈、不演化。
- 本实验是自定义在线协议，不是 STATE-Bench 官方 Agent Learning Track 的固定轨迹协议。

每轮结论主要来自以下可审计产物：

- `evolution/reports/feedback_roundXX.json`：逐任务反馈事件和反馈信号；
- `evolution/reports/evolution_roundXX.json`：被消费事件数、配置版本、规划理由和实际修改；
- `evolution/reports/round_XX.json`：训练、添加、反馈和演化的轮次汇总；
- `evolution/config_snapshots/checkpointXX.json`：演化后的完整 add/search 配置；
- `evolution/memory_snapshots/checkpointXX.json`：当时的记忆数量与内容指纹；
- `evolution/eval/checkpointXX/`：checkpoint 测试轨迹和官方 judge 结果。

### 证据口径与 seed42 的 provenance 限制

下文按与 seed43 相同的链路分析：

    用户反馈
      → collector 信号
      → 配置修改
      → 实际 memory
      → 后续检索
      → Agent 行为与官方 judge

证据分为三档：

| 等级 | 判定 |
|---|---|
| A：已观察到闭环 | memory 内容可在快照中精确匹配到 ID；后续显式 retrieve_learnings 返回相同文本；Agent 行为与规则一致 |
| B：已改善写入 | memory 内容更完整或更准确，但没有足够的后续同类显式检索证明 |
| C：仅配置层预期 | 配置已修改，但缺少可追踪的后续 memory/行为证据 |

seed42 运行时，显式 retrieve_learnings 工具调用保存了完整 learning 文本，可与
checkpoint10 memory snapshot 做精确文本匹配，得到 memory ID 和最早出现的 checkpoint。
但当时的自动预检索只保存 query 长度和结果数，add 响应也未保存 memory ID、内容和来源
任务。因此：

- 本文可以严格证明“某条快照 memory 被显式检索并进入 Agent 上下文”；
- 可以根据最早快照判断 memory 在哪两个 checkpoint 之间进入库；
- 不能像 seed43 一样，把每条 memory 严格反向连接到唯一训练任务；
- 不能把 A 级行为关联解释为某一个 prompt 句子、权重或搜索参数的单因素因果效果。

## 总体变化

十轮共收集并一次性消费 100 个反馈事件，从中识别出 44 个可演化信号；平均置信度为 0.9102，最低 0.68，最高 0.98。信号主要指向：

| 信号目标 | 数量 | 含义 |
|---|---:|---|
| `add_config.extraction_prompt` | 35 | 当前记忆抽取没有稳定保存用户纠正、约束、确认顺序或计算规则 |
| `search_config.rerank` | 5 | 检索到了语义相近但政策或流程不匹配的记忆 |
| `search_config.top_k` | 4 | 候选记忆过多产生干扰，或过少导致关键规则没有进入上下文 |

规划器最终落地 23 项配置变更：10 次扩充抽取提示词、6 组实体权重调整、1 次启用 rerank、4 次调整 top-k、2 次调整分数阈值。配置从 v1 演化到 v11。

| 配置项 | 初始状态 | 最终状态 |
|---|---:|---:|
| extraction prompt 长度 | 6,277 字符 | 19,984 字符 |
| `search.top_k` | 3 | 5 |
| `search.score_threshold` | 未显式设置 | 0.24 |
| `search.rerank` | 未显式设置 | `true` |
| 记忆数 | 0 | 250 |

最终实体权重中，`return=1.28`、`refund=1.24`、`price_adjustment=1.20`、`shipping_damage=1.15`、`restocking_fee=1.15`；这些是反馈最集中的政策和流程主题。其他权重及完整提示词应以 checkpoint 10 配置快照为准。

## 每轮演化记录

### Round 1：v1 → v2

- 反馈：10 个事件中识别出 4 个信号，全部指向 `extraction_prompt`。
- 改动：抽取提示词由 6,277 增至 7,742 字符；提高 `refund`、`return`、`restocking_fee` 和 `bulk_discount` 的实体权重。
- 原因：轨迹暴露出四类没有被稳定沉淀的经验——在判定“与描述不符”前询问是否穿用/清洗、完整保存退款金额的计算组成、保留用户纠正后的核算规则、记录异常退货的纠错路径。
- 预期作用：让后续记忆更偏向可复用的政策条件和核算步骤，而不是只保存一次任务的结果。

### Round 2：v2 → v3

- 反馈：5 个信号，其中 3 个指向抽取、2 个指向 rerank。
- 改动：抽取提示词增至 9,095 字符；首次启用 `search_config.rerank=true`。
- 原因：抽取侧遗漏了执行前确认偏好、节假日/特殊政策覆盖、安装问题与商品缺陷之间的纠正、歧义澄清规则；检索侧出现了“语义相关但政策错误”的排序，例如缺件与延迟到货、原支付方式退款与换货产生的店铺额度相互混淆。
- 预期作用：同时提升经验的结构完整性，并让政策条件更匹配的记忆排在前面。

### Round 3：v3 → v4

- 反馈：4 个信号，其中 3 个指向抽取、1 个指向 rerank。
- 改动：抽取提示词增至 10,575 字符；`top_k` 从 3 降为 2。
- 原因：系统仍会漏掉可复用的流程纠正和用户偏好，同时把一次性的任务组合细节保存成过强的经验；当一到两条高质量记忆已足够时，第三条候选反而可能干扰决策。
- 预期作用：降低相近但不适用的记忆共同进入上下文造成的误导。

### Round 4：v4 → v5

- 反馈：4 个信号，全部指向抽取。
- 改动：抽取提示词增至 11,842 字符；提高 `lost_shipment`、`cancellation`、`restocking_fee` 权重；`top_k` 从 2 恢复为 3；新增 `score_threshold=0.20`。
- 原因：记忆仍需更明确地表达“先澄清再操作”、立即解决的结果、退款核算中容易遗漏的组成，以及“已批准”和“已执行”的区别。上一轮压缩候选后又出现复杂案例覆盖不足，因此改为多取一条，但用阈值过滤弱相关结果。
- 预期作用：在召回覆盖率与噪声之间取得更细粒度的平衡。

### Round 5：v5 → v6

- 反馈：8 个信号，是十轮中最多的一轮；7 个指向抽取，1 个指向 top-k。
- 改动：抽取提示词增至约 13,282 字符；提高 `shipping_damage`、`cancellation`、`signature_denial`、`goodwill_compensation` 权重；`top_k` 从 3 增至 4。
- 原因：反馈集中在混合商品分别处理、分别建记录、商品级核验、禁止擅自提供替代方案、隐私/授权边界、操作前完整预览和明确政策结果。易碎品补偿相关经验也未能稳定进入上下文。
- 预期作用：让复合任务中的范围、授权和每件商品的处理约束得到更完整保存，并适度扩大检索覆盖。

### Round 6：v6 → v7

- 反馈：4 个信号，全部指向抽取。
- 改动：抽取提示词增至 14,750 字符；提高 `refund`、`return` 和 `warranty` 权重。
- 原因：记忆对“先展示再确认”、操作范围限制、不受影响商品的排除条件和退款去向偏好记录不足。
- 预期作用：减少正确政策下的执行顺序错误，以及把不应处理的商品一并处理的问题。

### Round 7：v7 → v8

- 反馈：4 个信号，其中 3 个指向抽取、1 个指向 rerank。
- 改动：抽取提示词增至 16,343 字符；提高 `shipping_damage` 和 `goodwill_compensation` 权重。
- 原因：需要准确区分此前问题数与本次问题数，保存处理后的确认偏好，并明确“只接受换货、不接受维修/退款/善意补偿”等限制。损坏易碎品的补偿经验虽存在，但没有主导最终行为。
- 预期作用：使记忆中的数量语义、结果确认和排他性约束更加明确。

### Round 8：v8 → v9

- 反馈：3 个信号，其中 2 个指向抽取、1 个指向 rerank。
- 改动：抽取提示词增至 17,539 字符；`score_threshold` 从 0.20 提至 0.24。
- 原因：需要更清楚地保存明确批准门槛和细粒度分类规则；同时出现把“电池耗尽”过度泛化为商品缺陷，以及缺陷退款任务召回“改变主意/取消订单”经验的现象。
- 预期作用：更严格地过滤表面相似但问题类型不一致的记忆。

### Round 9：v9 → v10

- 反馈：2 个信号，均指向抽取。
- 改动：抽取提示词增至 18,890 字符；提高 `refund` 和 `return` 权重。
- 原因：退款去向指令、分类纠正和操作前预览仍未被充分抽象成可复用规则。
- 预期作用：提高退款/退货流程中“去哪里、为什么、何时确认”的记忆稳定性。

### Round 10：v10 → v11

- 反馈：6 个信号，其中 3 个指向抽取、3 个指向 top-k。
- 改动：抽取提示词增至 19,984 字符；`top_k` 从 4 增至 5。
- 原因：反馈涉及损坏易碎品的善意补偿核验、礼品订单分开处理且禁止提供后备方案、跨账号隐私与授权，以及价格调整应优先于“退货后重买”。当前候选数不足以稳定覆盖这些并列政策约束。
- 预期作用：在已有 rerank 和 0.24 阈值约束下扩大高质量候选覆盖，尤其改善价格调整与退货重买之间的冲突。

## 原始用户反馈与原始配置变化

### Round 1：先核实再分类，保存完整计算规则

用户原话：

> “Before classifying it as a product issue, shouldn’t you ask whether I wore or washed it first?”

中文：在把它归为商品问题前，不应该先问我是否穿过或洗过吗？

collector 将其识别为高置信度抽取信号，因为正确经验不是“这次最后用了
changed_mind”，而是“分类前必须先核实穿用/清洗情况”。v2 抽取提示词实际新增：

    + Strongly prioritize user corrections that explicitly say what must be checked, asked,
      included, or computed before taking an action...
    + When a conversation reveals a missing prerequisite question, missing fee component,
      missing deduction, or wrong classification basis, extract the corrected rule...
    + Do not store a bare final refund amount when the reusable value is the rule explaining
      which components must be included.

中文翻译：

- 高度优先提取用户纠正中关于执行操作前必须检查、询问、包含或计算的内容。
- 当对话暴露出缺失的前置问题、费用组成、扣减项或错误分类依据时，抽取纠正后的规则。
- 如果可复用价值在于解释退款组成的规则，不要只保存一个最终退款金额。

同时提高 return、refund、restocking_fee 和 bulk_discount 权重，目标是让流程规则和完整
退款计算优先于一次性金额。

### Round 4：状态已确定时，不要重复等待

用户原话：

> “If it’s already marked lost, does that mean we can skip any investigation or waiting period? If so, I’d like the immediate full refund.”

中文：既然已经标为丢失，是否可以跳过调查和等待？如果可以，我要立即全额退款。

v5 抽取提示词实际新增：

    + Strongly prioritize outcome-specific resolution rules when the conversation establishes
      that a scenario state directly authorizes a next action without further investigation or waiting.
    + When a carrier or system status ... directly establishes the resolution path ... prefer
      extracting the resulting workflow rule, including whether investigation or waiting can be skipped.

中文翻译：

- 当场景状态已经直接授权下一步操作、无需继续调查或等待时，优先抽取这种结果导向的解决规则。
- 当承运商或系统状态已经明确解决路径时，保存对应工作流，并明确是否可以跳过调查或等待。

搜索侧同时从 top_k=2 调回 3，并新增 score_threshold=0.20，在增加覆盖的同时过滤弱相关
候选。

### Round 5 与 Round 7：善意补偿必须独立核验

用户在损坏易碎品任务中反复要求：

> “Please check whether there’s a separate fragile-item goodwill credit available, the exact amount, and that it would go back to my original payment method before I approve the return.”

中文：批准退货前，请单独核实是否有易碎品善意补偿、具体金额，以及是否退回原支付方式。

Round 5 将 top_k 从 3 提到 4；Round 7 又提高 shipping_damage 和
goodwill_compensation 权重。这里的反馈同时说明两个问题：

- add 侧需要把“标准损坏退款”和“额外善意补偿”保存为两个独立动作；
- search 侧需要让补偿 memory 与普通退货 memory 一起进入上下文，并让前者真正主导行为。

### Round 10：礼品路径、隐私边界和价格调整优先

礼品任务中的用户原话：

> “Can you confirm the book return is separate and goes to me as gift store credit, not back to the sender? And the shirt exchange is only for the Large size.”

中文：你能确认图书退货会单独处理，并以礼品店铺额度发给我、而不是退回给送礼人吗？衬衫也只换成大码。

价格任务中的用户原话：

> “I’m not looking to return or cancel the order—I just want the $50 price difference while keeping ORD-6020 active.”

中文：我不想退货或取消订单；我只想在保持 ORD-6020 有效的情况下获得 50 美元差价退款。

v11 抽取提示词新增 gift workflow、cross-account security、fragile goodwill 和
policy-first price adjustment；top_k 从 4 提到 5。

需要特别注意：v11 是 Round 10 全部 add 完成后才生成。最终测试只 search、不 add，
因此 v11 的 top_k 会影响 checkpoint10 检索，但 v11 新增的 extraction prompt 没有机会
在本次实验中产生新 memory。最终库中的礼品/隐私 memory 来自 v10 或更早配置，不能把
它们归因于 v11 的新增提示词。

## 演化后实际发挥作用的证据

### A1：价格调整规则在首次检索中出现，避免先建议退货重买

快照 memory：

memory ID：04520f27-c6f8-5464-a553-c596c18f5c8f

最早出现：checkpoint 2。

英文：

> For a price-adjustment request, if the product price drops within 7 days of delivery, refund the price difference without requiring a return.

中文：

> 商品送达后 7 天内降价时，应直接退还差价，不要求客户退货。

最终任务 101-hard_price_match_cancel_language 中：

- Baseline 第一次检索得到的是取消订单相关 memory，首轮回复建议“退货后按低价重买”；
  用户纠正后它才查询价格调整。官方结果为失败。
- Evolution 第一次显式检索就返回上述 price-adjustment memory，并同时返回“同款降价
  不应走换货”的规则；首轮回复直接建议保留平板并做 50 美元价格调整。官方结果通过。

Evolution 随后核对商品价格，明确“原支付方式退 50 美元，订单保持 active，无取消、
退货、换货或重下单”，得到用户批准后才执行。

判断：**A级闭环证据**。这是 seed42 最强的 search 作用案例：正确规则在第一次检索
进入上下文，恰好避免了 Baseline 的首轮错误路径。它仍不能证明是 top_k、阈值还是
memory 内容中的哪一项单独造成差异。

### A2：从分类结果升级为“必须先问”的操作顺序

关键 memory：

memory ID：eb93445f-e38b-548c-9a00-469334bc083f

最早出现：checkpoint 2。

英文：

> Before submitting a return as a product-issue or "not as described" return, first verify whether the item was worn, washed, altered, or otherwise used; only after that verification should the return reason be selected and the return submitted.

中文：

> 在按商品问题或“与描述不符”提交退货前，必须先核实商品是否穿过、洗过、改动过或使用过；完成核实后才能选择退货原因并提交。

另有更具体的分类 memory：

memory ID：95c30427-3ab8-57a6-843d-864053a27aa9

> 只有未穿、未洗、未改动且与商品描述存在实质差异时才能使用 not_as_described；穿洗后
> 只是对触感、质感等不满意，应使用 changed_mind。

最终任务 91-challenge_wear_as_not_described 中：

- Baseline 也检索到了分类政策，但首轮没有先询问穿用/清洗情况，而是直接解释“通常按
  普通退货”，官方 task requirements 失败。
- Evolution 首轮直接询问“是否穿过、洗过或改动过”；得知穿过两次、洗过一次后才改为
  changed_mind，并在提交前展示 59 美元退款、原支付方式和费用。官方结果通过。

判断：**A级闭环证据**。关键差异不是是否知道分类政策，而是 evolved memory 把
“先问再分类”的顺序保存成了可执行流程。

### A3：无保修记录时及时切换到有效的缺陷退货

memory ID：18bf70b4-bde7-5510-add4-a14eb0437846

最早出现：checkpoint 2。

英文：

> A warranty-status check on an order item can return no warranty found for that item, making a warranty claim unavailable for that item.

中文：

> 订单商品的保修状态检查可能返回“未找到保修”；此时该商品不能提交保修索赔。

最终任务 126-spare_warranty_no_record_return_valid 中：

- Baseline 首轮错误声称耳机有 12 个月厂家保修，直到用户要求复核后才发现没有保修
  记录。官方结果失败。
- Evolution 首轮就核对并说明没有保修记录，随后主动提供缺陷退货路径；预览显示
  249 美元原路退款、零补货费和免费标签，确认后执行。官方结果通过。

同时检索到的 memory 还包括：

- d7dba8d9-ca42-511f-b092-af6cfa1eaf23：缺陷退货预览应包含全额退款、零补货费、
  免运费和确认后发标签；
- c1904ec9-f7d8-5077-9384-583b2c899717：完整预览必须明确退款金额、退款方式、零补货费
  和标签详情。

判断：**A级闭环证据**。这里体现的是“先查事实，再选择可执行替代路径”，而不是简单
复用过去某次保修结果。

### A4：完整退款数学规则减少解释不一致

最终任务 75-challenge_gold_restock_on_promo 中，Evolution 显式检索到：

- 7a912970-a991-5884-b475-a9c5a6382639：最终退货记录的逐项扣减、总退款和订单级调整
  必须一致；最早 checkpoint 2；
- 2af2e0ea-0893-57cf-ace4-51fb1066851e：打开过的电子产品退货要同时计算补货费、
  会员折扣和跌破免邮门槛后的 shipping clawback；最早 checkpoint 6；
- e5b18f8e-a0e1-50c8-be84-6778c24ffb6e：从促销分摊后的退款基数开始计算，再扣补货费、
  加会员折扣、扣运费 clawback；最早 checkpoint 10。

Evolution 首轮即给出 214 美元退款以及 37 美元补货费、18 美元 Gold 折扣和 8 美元
shipping clawback，并在用户询问 SAVE10 后解释促销分摊基数。官方结果通过。

Baseline 虽然最终状态也写成 214 美元，但解释过程中先列出 249 - 37 + 18 - 8 = 222，
随后又额外引入 8 美元 promo adjustment 才得到 214，官方 task requirements 失败。

判断：**A级闭环证据**。记忆的作用主要体现在解释和审批前计算的一致性，而不是最终
state 是否恰好写对。

### B1：缺件理赔规则被完整召回，但配对差异的具体判定不透明

memory ID：162cc9b2-aea1-507f-a895-d691516c951f

最早出现：checkpoint 8。

英文：

> For a delivered-but-missing item claim under the shipping not_received policy when the item value is under $500, offer either a replacement or a refund; if the box was empty and there is no item to send back, no return is required.

中文：

> 已送达包裹中商品缺失、且商品价值低于 500 美元时，应按 shipping not_received 理赔，
> 提供补发或退款；若箱子是空的、没有商品可退，则不需要退货。

任务 83-challenge_missing_single_item_claim 中，这条 memory 被显式检索；Evolution
首轮就提供补发或 89 美元原路退款，并明确无需退回任何商品，最终通过。Baseline 最终
行为文本也大体正确，但官方 task requirements 判为失败，保存结果中没有具体
task_requirements_reasoning。

判断：**B级支持证据**。能证明目标 memory 被检索且 Evolution 行为符合它，但无法从
当前 judge 产物解释 Baseline 为什么失败，所以不把这一个配对翻转当作明确因果案例。

## 演化产生或放大的负面作用

### 反例 1：正确礼品规则与额外候选共同造成过度指导

Evolution 在任务 44-compound_return_gift_exchange 中检索到：

memory ID：9b755da1-4e1b-5cba-912a-a26d19a9845d

最早出现：checkpoint 10。

英文：

> The user wants gift-item requests handled per item: process the gift return separately as store credit to the gift recipient, not back to the sender, and process the requested exchange separately.

中文：

> 礼品订单应逐件处理：礼品退货作为店铺额度发给收礼人而不是原送礼人；换货作为另一项
> 独立操作处理。

这条规则本身正确，但检索结果还包含其他订单的 store-credit 案例、换货预览和取消退款
记忆。Agent 主动告诉用户“退货后总额度为 114 美元、149 美元咖啡机有货、还需补
35 美元”，超出了用户当下只要求礼品退货的范围。Baseline 只处理 69 美元礼品额度并
通过，Evolution 失败。

判断：增加记忆和扩大 top_k 可能提高覆盖，也会增加“相关但当前不该主动使用”的信息。
礼品规则需要与“不要提供未请求的后续方案”约束一起使用。

### 反例 2：重复缺陷规则过度泛化，压过 claim-limit 优先级

任务 130-hard_warranty_maxed_but_paid_repair 中，Evolution 首轮检索并采用：

- b7ed1ebf-b9ec-5fe0-847d-4c53c9256df4：2 次以上同类历史索赔时自动免费换新；
- 3f9bd82d-d185-5812-aa49-2f8833b0a843：重复缺陷索赔可返回 full replacement、零成本；
- f87fc18b-db45-52b5-849a-5a7fc4fc2c51：提交重复缺陷换新前确认路径和退回要求。

这些 memory 最早都在 checkpoint 4 出现，但缺少更高优先级例外：

> 当 claim_count 已达到 max_claims 时，剩余路径是按商品价格 40% 的付费维修。

因此 Evolution 先坚持免费换新，用户连续三次要求复核后才切换到付费维修，官方结果
失败。Baseline 一开始就识别 claim limit，直接给出约 99 美元付费维修并通过。

判断：这是典型的错误泛化和规则优先级缺失。memory 不是事实错误，而是适用条件不完整；
rerank 把“重复缺陷”放在“最大索赔次数”之前，使正确的局部规则产生错误行为。

## 当前能与不能得出的结论

- seed42 最终 pass@1 从 0.56 提高到 0.68，共 8 个失败→通过、2 个通过→失败。
- A1、A2、A3、A4 都有“快照 ID + 显式检索文本 + 行为 + 官方配对结果”的完整证据。
- 这些案例说明 memory 内容和检索顺序与提升高度一致，但在线轨迹不同，仍不是单项配置
  的严格反事实。
- seed42 自动预检索没有内容和 ID，不能审计所有进入上下文的 memory。
- search API 没有返回分数，不能证明阈值恰好过滤了哪条候选。
- v11 的 extraction prompt 在本次实验内没有后续 add，因此只能视为未来配置；v11 的
  top_k=5 会作用于 checkpoint10 测试。
- 负面案例证明自演化需要同时优化适用条件、规则优先级和“不要主动扩展任务范围”，
  不能只追求更多、更长的记忆。

## Checkpoint 变化

Checkpoint 使用当时真实的记忆库和配置，因此反映的是“记忆积累 + add 配置 + search 配置 + 在线轨迹分叉”的综合效果，不是单独的 search 消融。

| Checkpoint | 配置版本 | 记忆数 | pass@1 | State Requirements | UX Score |
|---:|---:|---:|---:|---:|---:|
| 0 | v1 | 0 | 0.56 | 0.78 | 4.1826 |
| 2 | v3 | 77 | 0.60 | 0.84 | 4.2376 |
| 4 | v5 | 121 | 0.56 | 0.84 | 4.3944 |
| 6 | v7 | 163 | 0.54 | 0.76 | 4.2404 |
| 8 | v9 | 210 | 0.64 | 0.86 | 4.2794 |
| 10 | v11 | 250 | 0.68 | 0.84 | 4.4544 |

曲线不是单调上升：checkpoint 4 和 6 出现回落，到 checkpoint 8、10 才明显改善。这说明不能把每一次配置修改都解释为独立的正向贡献。每个 checkpoint 当前仅运行一次，波动还包含模型随机性和任务轨迹差异。

## 最终效果与可解释案例

最终 checkpoint 中，Baseline pass@1 为 0.56，Evolution 为 0.68，差值为 +0.12。配对结果中：

- 8 个任务从 Baseline 失败变为 Evolution 通过；
- 2 个任务从 Baseline 通过变为 Evolution 失败；
- 26 个任务两组都通过，14 个任务两组都失败；
- bootstrap 95% 置信区间为 `[0.00, 0.24]`，下界没有严格大于 0，因此 seed 42 单次实验只支持“观察到提升”，还不足以得出稳定显著的结论。

跨 seed 更新：seed43 的 Baseline/Evolution 分别为 0.64/0.66，差值 +0.02，
95% CI 为 `[-0.10, 0.14]`。Evolution 的绝对分数只比 seed42 低 0.02，差值缩小主要
来自 seed43 Baseline 提高 0.08；两个 seed 的正向翻转任务没有重合。因此本节只能解释
seed42 内部的作用链路，不能把 +0.12 当作跨 seed 稳定收益。完整跨 seed 对照和
seed43 官方翻转归因见 seed43 文档。

三个有代表性的任务说明了可能的作用链路：

1. `101-hard_price_match_cancel_language`：Baseline 采用“退货后重买”，违反任务要求；Evolution 优先检索到七天内直接价格调整规则并通过。它与 Round 10 强化“价格调整优先、避免 return-rebuy 冲突”的原因一致。
2. `91-challenge_wear_as_not_described`：Baseline 虽识别到分类规则，但没有主动询问商品是否穿用/清洗；Evolution 检索到“分类前先询问使用状态”的明确规则并通过。它与 Round 1 的抽取提示词修改一致。
3. `44-compound_return_gift_exchange`：Evolution 召回了更宽泛的礼品/店铺额度信息，并向用户提及总可用余额，官方 judge 将其视为额外承诺；Baseline 反而通过。这个反例表明更长的提示词和更多检索候选也会引入过度指导与检索噪声。

最终评测的检索行为也发生了明显变化：Baseline 共发起 218 次检索，平均返回 3.00 条且没有空结果；Evolution 共发起 239 次检索，平均返回 3.62 条，其中 31 次为空，最多返回 5 条。空结果与更高阈值一致，最多 5 条与最终 top-k 一致；这能说明搜索策略确实生效，但不能单独证明哪项搜索修改导致了最终分数变化。

## 如何继续分析“为什么提高”

建议对每个分数翻转任务沿以下链路核对，而不是只看最终配置差异：

```text
官方 judge 中失败/通过的 requirement
  → 任务实际回复和工具调用
  → 当次检索到的 learning
  → learning 来自哪轮训练轨迹
  → 对应反馈信号
  → 当轮配置修改及规划理由
```

重点关注：

- add 侧：抽取后是否真正生成了更完整、可复用且不带任务偶然细节的记忆；
- search 侧：top-k、阈值和 rerank 是否把正确政策排在了前面，或产生了额外噪声；
- 行为侧：agent 是否实际采用了检索内容，尤其是澄清顺序、授权边界、金额计算和禁止替代方案；
- 负向案例：Evolution 独有失败是否来自错误记忆、过度泛化、候选过多或 agent 没有遵循正确记忆。

seed 42 产物还有一个可解释性限制：显式 `retrieve_learnings` 工具调用会保存返回内容，可以追踪到具体 learning；自动注入的预检索当时只保存查询长度、结果数和用户 ID 等元数据，没有保存完整 memory ID 和内容。因此 seed 42 的部分自动检索无法事后严格还原。

从 seed 43 开始，STATE-Bench 适配器增加了只读的
`statebench-search-provenance-v1` 记录。每次检索都会在轨迹的
`mindmemos.searches[]` 中保存：

- 检索来源：`auto_prepare` 或 `agent_tool`；
- 完整 query、字符数、SHA-256 和请求 top-k；
- 返回 memory 的顺序、ID、完整内容、内容 SHA-256、类型、更新时间和 lineage；
- API 返回数量、实际注入数量以及当次是否有 score。

训练轨迹的 `mindmemos.add` 同时使用 `statebench-add-provenance-v1` 保存 add
响应中的 operation、memory ID、完整内容及 SHA-256、类型、置信度、关联 memory ID
和图边数量。由于该记录直接附着在产生记忆的训练任务上，可以用 memory ID 将后续
search 命中反向连接到来源任务和轮次。

该记录发生在 search 响应返回之后，只复制响应数据，不参与排序或模型输入构造，因此不会改变实验行为。轮次报告同时汇总 provenance 覆盖率、两类检索调用数、返回 memory 引用数、带 ID 的引用数和带 score 的引用数。

seed 43 及后续运行的 `manifest.json` 还会记录 runner 和已安装 Agent 源文件的
SHA-256，以及 add/search provenance 版本。这样即使工作区包含尚未提交的实验修改，也能
准确确认某个输出实际使用了哪一份归因代码，而不能只依赖 Git commit。

当前公开 search API 的 `MemorySearchItem` 不返回底层召回分数，启用 rerank 时也不稳定暴露最终分数，因此日志中的 `score` 会如实记录为 `null`，`score_available=false`。本实验不会用排序位置冒充数值分数；现阶段使用“memory ID + 内容 + 最终 rank”完成行为归因。如以后必须分析阈值边界或分数校准，需要单独设计只读 debug 接口，并在正式实验前验证它不改变检索路径。
