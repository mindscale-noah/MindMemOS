ENTITY_DESCRIPTION_REWRITE_PROMPT = """
You are a memory editor. The description of an entity below has grown too long because new observations kept being appended. Rewrite it as ONE concise, self-contained description.

Rules:
- Keep the entity's stable identity facts and the most recent updates; drop redundant or low-value detail.
- Do not invent facts. Only compress what is given.
- Output at most {char_limit} characters, in the same language as the input.
- Return ONLY the rewritten description as plain text - no JSON, no tags, no explanations.

Entity name: {entity_name}
Entity type: {entity_type}
Current description:
{current_description}
"""
