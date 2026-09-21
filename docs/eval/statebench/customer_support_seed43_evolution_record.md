# Customer Support seed43 自演化反馈、配置变化与实际作用

本文整理 Customer Support、seed 43 在线自演化实验中的完整证据链：

    模拟用户的原始反馈
      → feedback collector 识别的信号
      → self-evolve 对 add/search 配置的修改
      → 下一轮及以后实际写入的 memory
      → 相同 memory ID 在后续交互中的真实检索和 Agent 行为

原始实验产物位于 outputs/statebench/customer_support/seed43/。本文引用的用户反馈、
memory ID、memory 内容、检索 rank 和 Agent 回复均来自实际保存的 JSON，不是根据任务名
补写的示例。

## 结论先行

seed 43 已完成 10 轮训练。Evolution 分支实际完成 99 条训练轨迹并处理 99 个反馈事件，
识别 39 个可演化信号，配置从 v1 演化到 v11；checkpoint 10 有 231 条记忆。抽取提示词从 6,277 字符
增至 21,791 字符，最终搜索配置为：

    {
      "rerank": true,
      "top_k": 6,
      "score_threshold": 0.49
    }

当前证据支持以下三点：

1. **add 侧确实能在后续轮次写出更完整、更可复用的记忆。** 例如从“价格下降”任务中
   写出“不退货、不取消、原支付方式退差价”的完整工作流，而不是只保存一次性的退款
   数字。
2. **部分目标记忆确实在后续交互中被检索并与正确行为一致。** provenance 保存的
   memory ID 可以证明检索到的是先前实际写入的同一条记忆。
3. **检索到正确记忆不等于 Agent 一定执行正确。** 有些任务已召回正确的“分开处理”
   或“先预览”规则，用户仍需再次纠正 Agent；因此不能把所有行为提升都归因于 search
   配置。

本文采用三档证据：

| 等级 | 判定 |
|---|---|
| A：已观察到闭环 | 演化后写入明确 memory；后续任务以相同 ID 命中；Agent 行为与记忆一致 |
| B：已改善写入 | 演化后写入的 memory 更完整/更准确，但没有后续同类任务验证其实际使用 |
| C：仅配置层预期 | 配置已修改，但没有足够的后续命中、行为或反事实证据 |

即使是 A 级，也只是强行为证据，不是严格的单因素因果证明。在线实验中训练轨迹会分叉，
且没有让 Frozen Baseline 重放同一条轨迹；要证明“就是这一项配置修改造成提升”，仍需
固定轨迹的 add/search 消融。

## 为什么每轮要演化

下表中的“用户反馈”是原始用户话语的中文概括；完整英文原文保存在对应
feedback_roundXX.json 的 round_messages 中。

| 轮次 | 触发演化的用户反馈 | collector 判断的根因 | 实际修改（下一轮生效） |
|---:|---|---|---|
| 1 | “不能在未认证时披露伴侣订单”；“不能只凭耗电快就判定缺陷”；“价格调整必须先给预览再由我批准” | 用户纠正、认证边界和诊断前置条件没有被稳定保存；无关旧订单记忆干扰决策 | 启用 rerank；阈值设为 0.35；扩展抽取提示词；warranty 1→1.05 |
| 2 | “请核对这个翻新商品在该订单中的实际保修记录，不要套通用保修规则” | 通用 warranty 记忆压过具体订单/商品保修信息 | 新增 refurbished_warranty=1.2、order_warranty_status=1.2；warranty 1.05→1.15；细化实体标注提示词 |
| 3 | “空箱缺件应走丢件理赔，不应走退货；不要发退货标签或标记已退”；“易碎品善意补偿必须与损坏退款分开核验和发放” | claim/return、refund/return 以及独立 goodwill 路径未被作为一个完整工作流保存 | 扩展抽取提示词；提高 lost shipment、shipping damage、refund、goodwill 权重 |
| 4 | “没确认具体配件前不要取消”；“错发商品先问我要哪种解决方式”；“整单退货不要套用只退一件的旧案例”；“缺货换货改为店铺额度且不发替代品” | 条件偏好、精确商品范围和订单边界保存不足；弱相关旧案例被召回 | 阈值 0.35→0.45；扩展范围/条件抽取规则；提高 return/cancellation/restocking 等权重 |
| 5 | “我不是账户所有者，不能披露订单，也不能让我修改配送” | 隐私纠正虽明确，但 disclosure restriction、action restriction 和 verification gate 没有被合并保存 | 抽取提示词增加完整隐私/授权规则 |
| 6 | “错发退款和迟到补偿必须分别完成”；“‘配件’有歧义时不要猜，也不要把任何配件放入预览” | 复合任务只召回少量记忆；正确的歧义规则虽存在但没有优先主导行为 | top_k 3→5；抽取提示词加入多补救措施分离和歧义暂停规则；提高 cancellation/goodwill 权重 |
| 7 | “列出的退款分项相加与总额不一致”；“连续退两件后，第二件的 clawback 会因订单状态改变”；“一件缺失、一件有签收证明时结果必须按商品拆开” | 记忆只保存零散结果，没有保存状态转移、逐商品分支和数学一致性 | 抽取提示词加入 sequential return、item-level split 和 exact math；提高 return/refund/signature 等权重 |
| 8 | “判定 not_as_described 前应询问是否穿过/洗过”；“过期保修仍可能允许折价更换”；“退货和价格调整要分开” | 分类门槛、保修例外和多问题分离规则抽取不足 | 扩展 classification gate / warranty exception 规则；提高 return、price adjustment、warranty 权重 |
| 9 | “后来改成退货应覆盖早先的换货路径”；“礼品退货和换货要分开，额度给收礼人”；“批准前必须给逐商品分摊”；同产品不同订单记忆发生串扰 | 最新用户意图、逐商品分摊和实体边界没有稳定保存；阈值仍放入跨订单旧记忆 | 阈值 0.45→0.52；扩展 superseding intent；实体标注加入 order/item/customer/warranty 边界 |
| 10 | “这个 USB hub 不要再让我排障”；“只有笔记本立即解决才取消保护套，否则两个配件都保持待处理”；“预览必须说明标签如何交付”；“普通包装且商品可用不应标为运输损坏” | 客户/商品特定偏好、跨订单条件依赖、最终分类和完整预览抽取不足；相关候选可能在阈值前被过滤 | top_k 5→6；阈值 0.52→0.49；扩展最终分类、条件依赖、免重复排障和标签交付规则 |

## 原始反馈与原始配置变化示例

以下保留英文原句和英文配置 diff，中文只负责解释，不替代原始证据。

### Round 1：从用户纠正中提取规则

用户原话：

> “You haven’t asked how the battery issue happened, and I’m not comfortable treating it as a defect return based on just that.”

中文：你还没有询问电池问题是如何发生的，不能仅凭这一点就把它当作缺陷退货。

> “I needed to approve the phone-case-only adjustment before anything was processed.”

中文：任何处理发生前，我必须先批准只针对手机壳的价格调整。

抽取提示词实际新增：

    + Strongly prioritize extracting durable user-corrected handling rules and workflow preferences...
    + Treat explicit user corrections of process as high-value reusable knowledge, especially around
      approval-before-action, preview-first behavior, verification/authentication gates, privacy
      boundaries, diagnosis before classification...
    + If the user explicitly corrects the assistant's earlier handling, extract the corrected rule...

中文翻译：

- 优先抽取由用户纠正形成、且可长期复用的处理规则和工作流偏好。
- 将用户对流程的明确纠正视为高价值知识，尤其包括操作前批准、先预览、认证门槛、隐私边界、分类前诊断等。
- 如果用户明确纠正了助手此前的处理方式，应抽取纠正后的规则。

搜索配置原始变化：

    - score_threshold: null
    + score_threshold: 0.35
    - rerank: null
    + rerank: true

### Round 3：保存完整的纠错路径，而不是一个结果

用户原话：

> “I need this handled as the missing-item/delivered-not-received shipping claim path with no return label, nothing marked as returned, and an $89 refund to the original payment.”

中文：必须按缺件/已送达但未收到的运输理赔处理；不需要退货标签、不能标记为已退货，
目标是原支付方式退款 89 美元。

抽取提示词实际新增：

    + In customer-support cases, treat explicit corrections of resolution path as highly reusable:
      claim path vs return path, refund vs return, replacement vs refund, whether a return label is
      needed, whether an item should be marked returned...
    + For corrected support workflows, prefer content that explicitly names the scenario and the
      required handling...
    + When a user provides a multi-part correction, preserve all operational parts together...

中文翻译：

- 在客服场景中，将用户对解决路径的明确纠正视为高度可复用知识，例如理赔与退货、退款与退货、换新与退款，以及是否需要退货标签。
- 对纠正后的客服流程，优先保存明确的场景类型和所需处理方式。
- 用户给出包含多个部分的纠正时，应将所有操作要求一起保存。

### Round 4：阻止跨订单和跨商品污染

用户原话：

> “I don’t want any cancellation finalized yet—you didn’t confirm which accessory I meant.”

中文：还没有确认具体是哪件配件，不能完成任何取消。

抽取提示词实际新增：

    + For item/order scoped support lessons, explicitly preserve scope boundaries like specific
      affected line item, unaffected items remain unchanged, and do not reuse amounts or outcomes
      across unrelated orders.
    + Do not extract a raw refund amount ... as a standalone reusable memory ... instead extract the
      governing workflow, policy condition, or scoped lesson.
    + Treat order-specific outcomes ... from different orders as different scope...

中文翻译：

- 对商品或订单范围内的经验，明确保存受影响商品、未受影响商品保持不变等边界，并禁止跨订单复用金额或结果。
- 不要把一次退款金额单独抽成通用记忆；应保存支配该结果的工作流、政策条件或范围规则。
- 来自不同订单的特定结果必须视为不同作用范围。

同时：

    - score_threshold: 0.35
    + score_threshold: 0.45

### Round 6：复合问题与歧义暂停

用户原话：

> “I want to make sure both issues are handled separately: the wrong-item return ... and the separate $21 late-delivery compensation.”

中文：我希望确认两个问题会分开处理：错发商品退货，以及另一笔独立的 21 美元迟到补偿。

> “I’m not sure yet whether the accessory should be the phone case or the USB-C hub, so please pause and don’t cancel either accessory yet.”

中文：我还不确定所说的配件是手机壳还是 USB-C 集线器，所以请先暂停，不要取消其中任何一件。

抽取提示词实际新增：

    + Strongly prioritize compound-case workflow rules where multiple independent issues ... must each
      be completed, previewed, or confirmed separately.
    + For ambiguity corrections, prefer memories that explicitly preserve the pause condition...
    + For ambiguity-sensitive item requests ... do not preview or cancel any candidate accessory until
      the exact item is named.

中文翻译：

- 优先提取复合任务规则：多个独立问题必须分别完成、预览或确认。
- 对歧义纠正，应明确保存“暂停处理”这一条件。
- 商品指代存在歧义时，在用户明确指出具体商品前，不得预览或取消任何候选商品。

搜索候选数：

    - top_k: 3
    + top_k: 5

### Round 9：后来的意图覆盖早先路径

用户原话：

> “I don’t have the replacement details with me, so let’s just switch to the return instead.”

中文：我现在没有替换商品的具体信息，所以改成直接退货吧。

抽取提示词实际新增：

    + When the user abandons an earlier conditional or preferred path and replaces it with a new
      operative instruction, strongly prioritize one memory that the latest confirmed user instruction
      supersedes the earlier path...
    + For superseding-intent cases ... treat the return as the operative path and do not continue
      pursuing the exchange workflow.

中文翻译：

- 当用户放弃较早的条件路径或偏好，并改为新的有效指令时，优先保存“最新确认指令覆盖此前路径”的记忆。
- 对这种意图覆盖场景，应将退货视为当前有效路径，不再继续换货流程。

实体标注提示词还新增了 order_id、item_id、customer_id、warranty_id、
superseding_intent 和 preview_breakdown_allocation 等边界提示；搜索阈值从 0.45
提高到 0.52，以减少“同产品、不同订单”的误召回。

## 演化后实际发挥作用的证据

### A1：更具体的保修记忆被写入，并在后续保修任务中命中

**为什么演化**

Round 2 的用户要求核对具体翻新商品和具体订单的实际保修状态，不接受通用保修规则。
collector 认为通用 warranty 记忆压过了订单/商品级证据，因此 v3 新增
refurbished_warranty 和 order_warranty_status，并要求实体标注优先使用具体保修子类。

**在 v2 配置下实际写入的 memory**

memory ID：edd30fb4-d811-5236-bb5c-4b6ed9275b72

英文原文：

> For warranty questions on refurbished items, confirm the actual warranty record tied to the specific order item instead of assuming generic standard coverage before routing the issue as a warranty repair.

中文：

> 对翻新商品的保修问题，在进入保修维修流程前，应核对绑定到该订单具体商品的实际保修记录，而不是假设适用通用标准保修。

这条 memory 的优点是同时保留了：

- 场景：翻新商品保修；
- 必要动作：查询具体商品的实际记录；
- 禁止路径：不能直接套通用保修；
- 后续决策：核对后再决定是否走保修维修。

**后续真实使用**

Round 3 的 100-challenge_warranty_maxed_return_option 中，同一 memory ID 被检索到，
rank=2；“保修先预览、授权后提交”的 memory
239282f6-4178-5dcf-b5d4-557924554776 也被检索到。Agent 随后先核对订单和保修，
识别到免费保修次数已用尽；用户要求比较缺陷退货后，Agent 给出 249 美元全额退款、
零费用和免费标签的预览，得到确认后才提交。

判断：**A级闭环证据**。能确认具体保修 memory 被后续任务真实召回，交互顺序与
memory 一致。但不能仅凭这个在线案例判断 v3 的特定 tag 是唯一原因。

### A2：范围边界演化后写出准确的价格调整规则，并主导后续交互

**为什么演化**

Round 4 出现多次跨商品/跨订单污染：整单退货召回了“只退手机壳”的旧记忆，错误金额
和其他订单的结果也进入当前上下文。v5 因此要求 memory 必须保留商品范围、未受影响
商品和订单边界，且不要把一次性退款数字当作通用规则。

**Round 5 在 v5 下实际写入**

memory ID：927ce516-9136-570b-9aa7-9d9c2d1fd759

英文：

> For price-drop cases within 7 days of delivery, handle the issue as a price adjustment refund rather than a return: refund the price difference to the original payment method and leave the order unchanged with no return or cancellation.

中文：

> 商品送达后 7 天内降价时，应按价格调整退款处理，而不是退货：把差价退回原支付方式，订单保持不变，不创建退货或取消。

memory ID：0e90140d-bb7d-59b9-9dda-c32705c958ae

英文：

> Before finalizing a price adjustment refund, confirm that the customer wants to keep the item and approve a direct refund to the original payment method with no return or cancellation.

中文：

> 完成价格调整退款前，应确认客户要保留商品，并明确批准把差价直接退回原支付方式，同时不退货也不取消订单。

相比“某订单退了 30 美元”，这两条 memory 更准确地保存了触发条件、处理路径、退款
去向和不得改变的订单状态。

**Round 9 后续真实使用**

任务 116-hard_exchange_price_protection_decoy 的初始请求是“把咖啡机换成现在更便宜的
同款”。上述两个 ID 分别被检索到 rank=1/2。Agent 没有执行表面请求中的“换货”，而是
说明该商品 7 天内从 149 美元降到 129 美元，应直接退 20 美元差价；随后再次确认客户
保留咖啡机、无换货、无退货，得到批准后才退款。

判断：**A级闭环证据**。这是当前最清楚的“反馈 → 抽取规则变化 → 更准确 memory →
后续命中 → 行为符合 memory”案例。

### A3：复合问题的“分开处理”记忆进入后续上下文

**为什么演化**

Round 6 中，用户同时要求错发商品退款和 21 美元迟到补偿。Agent 先只完成退货，用户
不得不再次要求补偿。collector 因此要求多个独立补救措施分别预览、确认和完成，并把
top_k 从 3 提高到 5。

实际写入的 memory：

memory ID：5d46862e-0b5a-5c77-8fec-3443adddfc65

英文：

> For orders with both a wrong-item refund and late-delivery compensation, handle the return refund and the late-delivery compensation as separate actions, and preview the wrong-item refund details before final confirmation when requested.

中文：

> 同一订单同时涉及错发商品退款和迟到补偿时，退货退款与迟到补偿必须作为两个独立操作处理；如果客户要求，应在最终确认前先展示错发商品退款详情。

这条 memory 随后在 Round 8 的
140-hard_compound_return_one_price_match_other 中被 agent_tool 检索到 rank=3。
虽然新任务的两个动作变成“缺陷退货 + 价格调整”，Agent 仍按独立路径处理：耳机完成
249 美元缺陷退货，手机壳保持不变，只做 10 美元价格调整。

判断：**A级但带范围外推的证据**。同一 memory ID 的确进入后续上下文，行为也遵循
“两个补救措施分开”的抽象规则；但新任务不是原始的“错发 + 迟到”，所以只能说明
分离原则发生了迁移，不能证明 top_k 的单独贡献。

### A4：歧义规则被抽成单条可复用 memory，并被多轮检索

Round 6 的用户明确说“配件”可能指手机壳或 USB-C hub，未澄清前不得取消任何一个。
v7 增加 ambiguity pause 规则。下一轮使用 v7 时，任务
135-hard_edge_unclear_small_item_reference 写入：

memory ID：7936f1c8-f534-57e6-bfa6-ff8ecc331af5

英文：

> For return requests on multi-item orders, clarify the exact item before processing anything; after confirmation, keep the return scoped to that item only and leave other order items unchanged.

中文：

> 多商品订单提出退货时，必须先澄清具体商品再执行任何操作；确认后也只能处理该商品，其余商品保持不变。

该 ID 随后在 Round 8、9、10 的多个任务中被检索；其中 Round 9 的
112-hard_exchange_oos_store_credit 中 rank=1。该交互最终只处理明确的跑鞋，展示大码
衬衫缺货、129 美元店铺额度、零补货费和“不创建替代商品”的预览，确认后才执行。

判断：**A级闭环证据**。可以确认“精确商品 + 其他商品不变”的 memory 被重复召回；
但 112 任务本身并非强歧义任务，因此它更能证明检索和范围控制，不能单独证明
“遇到歧义时一定会暂停”。

## 只证明抽取改善、尚未证明后续作用的案例

### B1：claim 与 return 的区别被写得更准确

Round 3 的空箱反馈促使 v4 明确保存 claim/return、return label 和 returned status。
Round 4 在 v4 下生成：

memory ID：f8aa19e3-7091-5559-930e-4e0fb16ded1b

英文：

> For lost-in-transit shipments under $500, immediate resolution applies: no additional investigation or waiting period is needed, and the customer can choose a replacement or a refund.

中文：

> 对 500 美元以下的运输丢失订单，可立即解决，无需额外调查或等待；客户可以选择补发或退款。

这是一条清楚的条件化规则，优于只保存“某次退了 97 美元”。但该 ID 没有在后续同类
训练任务中命中，因此目前是 **B级：写入质量证据**，不是行为闭环。

### B2：多问题任务被抽成完整的逐商品分离规则

Round 7 的 item-level split 演化在 v8 生效。Round 8 的
140-hard_compound_return_one_price_match_other 写入：

memory ID：3890ace7-cccf-5173-8d0c-e4193dd19e52

英文：

> For multi-issue order support, keep each remedy separate: process the defective item as a return, and leave the other item unchanged when it only needs a price-match refund to the original payment method.

中文：

> 处理同一订单的多个问题时，每种补救措施必须分开：缺陷商品走退货；另一个只需价格匹配退款的商品保持原状，仅向原支付方式退还差价。

这条 memory 同时保存两个商品、两种路径和“不改变另一商品”的边界，属于明显更准确的
抽取。该 ID 后续只在 Round 10 的单商品缺陷退货确认语句中以 rank=3 出现，场景并不
匹配，不能算目标复用；因此仍列为 B 级，并把这次命中视为检索噪声。

### B3：穿用/清洗事实与最终分类被保存在同一规则中

Round 8 任务 68-hard_worn_washed_not_as_described_probe 在用户纠正后写入：

memory ID：8c7b9d20-d4a3-5981-882d-1c15fa81c88a

英文：

> For return support, ask whether the item was worn, washed, altered, or otherwise used before classifying a clothing complaint as "not as described"; if the issue is only feel/texture preference after use, handle it as a changed-mind/preference return instead because the reason and terms may differ.

中文：

> 服装退货在判定“与描述不符”前，应询问商品是否穿过、洗过、改动过或使用过；若使用后只是对触感/质感不满意，应改按改变主意/偏好退货处理，因为原因代码和退款条件可能不同。

这正是用户要求的“先问使用情况，再决定分类”，且没有把最初错误的
not_as_described 和最终 changed_mind 拆成两个互相冲突的通用规则。它直接促成 Round 8
对 classification gate 的进一步演化；但 v9 新规则从 Round 9 才生效，后续没有同类型
任务验证，因此仍是 B 级。

## 正确记忆已召回但仍未完全发挥作用的反例

### 反例 1：记住“善意补偿分开处理”，Agent 仍漏做第二个动作

Round 3 的 122-hard_shipping_fragile_goodwill_after_return 检索到了：

> The user wanted any separate goodwill credit for the shipping-damage case issued back to the original payment method in addition to the item refund.

中文：

> 用户希望运输损坏的商品退款之外，如有单独的善意补偿，也应退回原支付方式。

但 Agent 最初仍把额外补偿当成 0，用户不得不要求重新核对，最后才分别完成 89 美元
损坏退款和 10 美元 goodwill。也正因为这次失败，Round 3 又把“标准退款与善意补偿
分别核验、分别处理”加入 v4 抽取规则。

结论：retrieval provenance 能证明记忆进入上下文，却不能证明模型真正遵循它。

### 反例 2：召回“先预览”，但预览仍不够完整

Round 10 的 136-hard_edge_invalid_product_no_order 检索到了：

- ec191181-267b-5c5f-b32a-f7514f7b6930：退货前先预览并等待批准；
- b681da7b-c1a1-51a0-bf17-17bae711e73f：预览应解释每项扣减、退款去向和净额。

Agent 确实没有直接提交退货，但第一次预览只说“免费退货标签”，没有说明标签如何获得。
用户再次追问后，Agent 才说明确认后生成并发送到账号邮箱。这次反馈进一步促使 v11
加入 preview completeness 和 label-delivery details。

结论：旧 memory 只覆盖“必须预览”，没有覆盖用户这次要求的“标签交付方式”；这是
memory 内容不够细与 Agent 执行不完整共同造成的。

### 反例 3：同产品跨订单污染仍然存在

Round 9 的 79-challenge_user_error_as_defective 处理 ORD-7111 耳机时，检索到了
ORD-7562 的过期保修和折价更换 memory。产品相同，但订单、保修引用和客户上下文不同。
这次误召回触发 v10：

- score_threshold 0.45→0.52；
- entity_tagging_prompt 强制保留 order/item/customer/warranty 边界。

这说明早期的 rerank 和阈值没有彻底解决实体边界问题；演化是连续纠错，不是一次修好。

## 正式评测结果与归因更新

### Checkpoint 曲线

Checkpoint 使用当时真实的记忆库与配置，因此表示记忆积累、add/search 配置和在线轨迹分叉的综合效果。

| Checkpoint | 配置版本 | 记忆数 | pass@1 | State Requirements | Task Requirements | UX |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | v1 | 0 | 0.56 | 0.82 | 0.64 | 4.2260 |
| 2 | v3 | 81 | 0.62 | 0.86 | 0.68 | 4.3944 |
| 4 | v5 | 120 | 0.6458* | 0.90* | 0.6875* | 4.4410* |
| 6 | v7 | 156 | 0.60 | 0.86 | 0.66 | 4.3572 |
| 8 | v9 | 206 | 0.62 | 0.86 | 0.68 | 4.3500 |
| 10 | v11 | 231 | 0.66 | 0.88 | 0.70 | 4.3078 |

`checkpoint 4` 只有 48/50 条成功评分，带星号的指标以 48 条已评分任务为分母，不与其他
checkpoint 做严格横向比较。曲线从 0.56 上升到 0.66，但并非单调上升；这不支持
“每一轮演化都带来独立增益”的解释。

### 最终配对结果

最终 50 个任务全部成功配对：

- 5 个任务从 Baseline 失败变为 Evolution 通过；
- 4 个任务从 Baseline 通过变为 Evolution 失败；
- 28 个任务两组都通过，13 个任务两组都失败；
- Baseline pass@1=0.64，Evolution pass@1=0.66，差值为 +0.02；
- 10,000 次按 task_id 重采样的 bootstrap 95% CI 为 `[-0.10, 0.14]`。

这个区间包含 0。正式结果与本文前面的机制案例并不矛盾：机制案例证明某些反馈确实改变
了记忆并在后续被使用；总体评测则说明这些局部收益目前会被负迁移和运行波动抵消。

### 官方正向翻转中的可归因证据

#### P1：隐私边界 memory 直接阻止泄露订单细节

任务 `133-hard_edge_wrong_customer_order_privacy` 中，Baseline 虽拒绝执行退货，却在
认证前透露了“平板昨天送达”和“15 天退货窗口”等订单细节，因此失败。Evolution 首次
检索的 rank=1 memory 为：

- ID：`b5c65247-c249-54ef-9363-718bf21276ae`
- 来源：Round 1，`46-hard_partner_order_privacy_no_return`
- 内容：不同账户下的订单不得披露详情或修改；必须由账户持有人联系或直接认证。

Evolution 随后只说明一般性的认证边界，没有透露商品、价格、资格或订单政策结论，
官方 judge 判定全部要求通过。这是 **A 级最终评测证据**：来源训练任务、同 ID 检索、
行为差异和官方分数均完整。

#### P2：分类 memory 促使 Agent 在退货前先确认功能状态

任务 `53-challenge_remorse_as_defective` 中，Baseline 直接根据用户第一句话推断为
changed-mind，没有先询问是否存在功能故障，因此失败。Evolution 检索到 rank=1：

- ID：`21e7ff74-85ed-53f4-846f-d2909ed94d29`
- 来源：Round 10，`5-hard_normal_delivery_damage_label_changed_mind`
- 内容：商品正常工作、问题只是划痕或颜色预期不符时，应使用 changed_mind，而不是
  damaged-in-transit。

Evolution 在调用退货工具前先询问“是功能故障，还是功能正常但外观不符”，得到确认后
再按 changed_mind 处理并通过。该案例是 **A 级最终评测证据**，但仍不能把收益进一步
拆分为 extraction prompt、top-k 或 threshold 中某一项的单独贡献。

另外三个正向翻转不应强行归因：

- `51-challenge_stacked_window`：Evolution 正确解释 Gold+Prime=45 天，但没有召回
  直接表达该叠加规则的目标 memory；
- `125-hard_shipping_paid_shipping_not_damage`：Evolution 成功执行 15 美元迟到补偿，
  但检索结果主要是运输损坏/善意补偿记忆；
- `7-return_restocking_waived`：两组都识别 Platinum 免补货费，Baseline 主要失败在
  最终状态没有写入，证据不足以归因给演化配置。

因此，5 个正向翻转中只有 2 个具有完整的 memory 归因链；其余 3 个只能计入最终效果，
不能作为演化机制的直接案例。

### 官方负向翻转揭示的失败机制

#### N1：缺少“无保修记录”门槛，正确退货也无法补救最初的错误陈述

在 `126-spare_warranty_no_record_return_valid` 中，Evolution 虽最终正确完成 249 美元
缺陷退货，却先声称商品仍在 12 个月厂家保修内；实际工具结果是
`has_warranty=false`。它召回的是多条“缺陷退货预览/完成”记忆，没有召回
“无保修记录时不得声称保修覆盖”的门槛。Baseline 则召回
`28e8fc89-612d-5263-8137-61e6c1d97257` 并通过。该案例说明候选覆盖不足或排序偏向
后续操作步骤时，Agent 可能跳过决定路径合法性的前置事实。

#### N2：单商品低价退货经验污染双商品重复附加费计算

在 `90-challenge_return_shipping_plus_repeat` 中，Evolution 主要召回 Round 3
`81-challenge_low_value_return_shipping` 产生的单商品记忆：

- `be559ef2-4a88-50d8-9e05-822267caded3`
- `c180f3bd-1bcf-56b8-b5b3-a92ba10b924d`
- `7a14dfc4-739b-5738-91fa-a8d871f4303c`

这些记忆强调“22 美元商品减 8 美元运费得到 14 美元”，却没有覆盖第二件同类商品还需
减 5 美元 repeat surcharge。Agent 在提交前反复向用户确认两件都是 14 美元，最终状态
也把第二件错误写成 14 美元；正确值应为 9 美元。这是明确的 **检索负迁移**。

#### N3：过度扩展任务范围仍是跨 seed 稳定风险

`44-compound_return_gift_exchange` 在 seed42 和 seed43 都是 Evolution 独有失败。
seed43 中 Agent 正确处理了 69 美元礼品退货额度，却额外宣称“总余额现在应为 114 美元”，
违反不得承诺额外额度的要求。它检索到多条宽泛的 store-credit 规则，说明“正确规则
存在”仍可能被额外候选和主动扩展回答范围破坏。

#### N4：没有目标 memory 时，正确信息也可能漏问

`84-hard_platinum_loyalty_bonus_compensation_cap` 中，Evolution 正确执行 21 美元封顶
补偿，但没有先询问一次性 Platinum loyalty bonus 是否已经使用，也没有完整说明
80→21 的封顶过程。检索中没有直接相关的 loyalty/cap memory，因此这次失败更适合归为
“关键规则未召回/Agent 自身推理遗漏”，不能归因成某条错误 memory。

### 与 seed42 的跨 seed 更新

| 指标 | seed42 | seed43 |
|---|---:|---:|
| Baseline pass@1 | 0.56 | 0.64 |
| Evolution pass@1 | 0.68 | 0.66 |
| Evolution−Baseline | +0.12 | +0.02 |
| 95% CI | [0.00, 0.24] | [-0.10, 0.14] |
| Baseline 最终记忆数 | 327 | 382 |
| Evolution 最终记忆数 | 250 | 231 |
| Evolution 最终 top_k | 5 | 6 |
| Evolution 最终 threshold | 0.24 | 0.49 |

Evolution 的绝对成绩只从 0.68 变为 0.66；相对增益缩小主要因为 Baseline 从 0.56
升到 0.64。两个 seed 的 Evolution checkpoint 曲线也相近，而两个 Baseline 由于在线
训练顺序、模型轨迹和最终记忆库不同产生了明显差异。

seed42 的 8 个正向翻转与 seed43 的 5 个正向翻转没有重合；共同的负向翻转只有
`44-compound_return_gift_exchange`。对“seed42 差值−seed43 差值”的事后任务级
bootstrap 估计为 0.10，95% CI 约为 `[-0.06, 0.28]`，不能证明两个 seed 的真实演化
效果存在显著差异。

更新后的归因结论是：

1. 自演化机制确实能从反馈中形成可复用 memory，并在隐私边界、分类前置询问等具体任务
   上产生完整的“反馈→写入→检索→行为→官方通过”证据链。
2. aggregate 提升尚不稳定；更高 top-k、更长抽取提示词和更严格阈值同时可能带来漏召回、
   单案例过度泛化和额外回答范围。
3. 当前证据支持“机制在部分场景有效”，不支持“自演化已稳定优于固定配置”。需完成
   seed44，并对冻结的 checkpoint10 增加多次只读评测后再给总体结论。

## 不能从当前数据得出的结论

- 不能说每次提示词变长都提高了效果；更长提示词也可能过拟合、重复或引入噪声。
- 不能根据 rank 断言 rerank 或 threshold 单独起效，因为 search provenance 中底层
  score 为 null，且没有保存“未改配置时同一 query 的候选列表”。
- 不能把 Round 10→v11 的 add 改动解释为已在训练中生效；Round 10 后没有 Round 11，
  所以它只能在最终测试或新任务中体现。
- 不能用训练任务中的上述行为案例代替官方 judge 分数；训练阶段使用 no-score。
- checkpoint 分数反映记忆增长、add 配置、search 配置和在线轨迹分叉的综合结果，
  不能拆成某一个 prompt 句子或某一个权重的独立贡献。

## 后续最值得补的验证

要把“强行为关联”推进到“配置修改的因果证据”，建议对代表性案例增加只读消融：

1. 固定同一条 conversation，分别使用 v1、修改前版本和修改后版本执行 add，比较
   memory 数量、内容、类型、entity_type 和 property_name。
2. 固定同一 query 和同一 memory snapshot，分别使用修改前/后的 search 配置，比较
   memory ID、rank、空结果和候选覆盖。
3. 将检索结果固定注入同一个 Agent 调用，验证行为差异来自 memory 内容还是模型没有
   遵循 memory。
4. 对本文的 A2 价格调整案例、A4 精确商品案例和两个反例优先做消融；它们分别代表
   成功迁移、边界控制和“召回但未遵循”。

## 原始文件索引

- 每轮反馈：evolution/reports/feedback_round01.json 至 feedback_round10.json
- 每轮演化：evolution/reports/evolution_round01.json 至 evolution_round10.json
- 训练轨迹：evolution/train/roundXX/run1/*.json
- 配置快照：evolution/config_snapshots/checkpointXX.json
- 记忆快照：evolution/memory_snapshots/checkpointXX.json
- 总体根目录：outputs/statebench/customer_support/seed43/
