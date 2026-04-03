# Prompt for task parsing
TASK_PARSE_PROMPT = """
You are a task parsing expert. Given a user's task instruction, extract the following structured information:

Given a user task instruction and optional related memory context,
extract the following structured information:
1. Keys: the high-level keywords directly relevant to the user’s task.
2. Tags: thematic tags to help categorize and retrieve related memories.
3. Goal Type: retrieval | qa | generation
4. Memories: Provide 2–5 short semantic expansions or rephrasings of the task instruction.
   These are used for improved embedding search coverage.
   Each should be clear, concise, and meaningful for retrieval.

Task description:
\"\"\"$task\"\"\"

Context (if any):
\"\"\"$context\"\"\"

Return strictly in this JSON format:
{
  "keys": [...],
  "tags": [...],
  "goal_type": "retrieval | qa | generation",
  "memories": ["...", "...", ...]
}
"""


REASON_PROMPT = """
You are a reasoning agent working with a memory system. You will synthesize knowledge from multiple memory cards to construct a meaningful response to the task below.

Task: ${task}

Memory cards (with metadata):
${detailed_memory_list}

Please perform:
1. Clustering by theme (topic/concept/fact)
2. Identify useful chains or connections
3. Return a curated list of memory card IDs with reasons.

Output in JSON:
{
  "selected_ids": [...],
  "explanation": "..."
}
"""
