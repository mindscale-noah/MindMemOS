ENTITY_DESCRIPTION_REWRITE_PROMPT = """
你是一名记忆编辑。下面这个实体的描述因为不断追加新观察而变得过长。请把它改写为一段简洁、完整的描述。

规则：
- 保留实体的稳定身份事实和最近的更新；删除冗余和低价值细节。
- 不要编造事实，只压缩已有内容。
- 输出不超过 {char_limit} 个字符，与输入语言保持一致。
- 只返回改写后的描述纯文本——不要 JSON、不要标签、不要解释。

实体名称：{entity_name}
实体类型：{entity_type}
当前描述：
{current_description}
"""
