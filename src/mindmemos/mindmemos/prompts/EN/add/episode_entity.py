"""Episodic entity (name + description + search fields) generation."""

EPISODE_ENTITY_PROMPT = """
You are an episodic memory expert. For the following conversation, produce a single JSON object that defines the episode's title, factual summary, and searchable phrases.

Conversation timestamp: {conversation_timestamp}
Conversation content:
{conversation_text}

Speaker note: Lines may be formatted as `speaker=Name: ...` for named-speaker dialogue. Treat `Name` as the real speaker of that line; first-person statements in that line belong to `Name`, not automatically to the user. Use explicit speaker names whenever available.

Return a JSON object with exactly these fields:
{{
    "title": "A concise, descriptive title (10-20 words) that accurately summarizes the theme",
    "content": "A brief factual summary (2-4 sentences) of the main topic and key facts, written in third person",
    "search_fields": ["up to {max_fields} short search phrases (5-20 words each)"]
}}

## title & content requirements
1. The title should be specific and easy to search, including key topics, activities, and participant names.
2. The content should briefly summarize the main topic and key facts in third person. Keep it concise — do NOT reproduce the full conversation.
3. Focus on searchable elements: WHO did WHAT, WHEN, WHERE, and WHY.
4. Include all proper nouns: person names, place names, brand names, product names, book/movie titles.
5. Include key numbers, dates, times, quantities, and prices.
6. Use the dual time format for relative references: "relative time (absolute date based on {conversation_timestamp})".
7. Use specific names consistently rather than pronouns.
8. Remove conversational filler, greetings, and redundancy.
9. Preserve causal relationships and decision reasoning.
10. Content should be significantly shorter than the original conversation — a brief overview, not a detailed retelling.

## search_fields requirements
1. **Diversity**: Each field should capture a DIFFERENT aspect or angle of this episode. Avoid redundancy.
2. **Query-oriented**: Write phrases as a user would naturally search for this information.
3. **Comprehensive coverage**: Cover who, what, when, where, why, plus specific names, items, plans, and outcomes mentioned.
4. **Self-contained**: Each phrase should be independently meaningful without needing the others.
5. **Include key names**: Mention the main participants in the fields for disambiguation.
6. Group related information into higher-level summary phrases when the episode has more facts than the field budget allows.

Return only the JSON object, no other text.
"""
