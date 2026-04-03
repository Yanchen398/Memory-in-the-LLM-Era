AGGREGATE_PROMPT = """
You will receive two pieces of information:  
New Information is detailed, and Existing Information is a summary from {n_children} previous entries.  
Your task is to merge these into a single, cohesive summary that highlights the most important insights.
 Focus on the key points from both inputs.
 Ensure the final summary combines the insights from both pieces of information.
 If the number of previous entries in Existing Information is accumulating (more than 2), focus on summarizing more concisely, only capturing the overarching theme, and getting more abstract in your summary.
Output the summary directly.

[New Information]
{new_content}
[Existing Information (from {n_children} previous entries)]
{current_content}

IMPORTANT! Don't output additional commentary, explanations, or unrelated information. Provide only the exact information or output requested.
[Output Summary]
"""

SEGMENT_SUMMARY_PROMPT = """
Your task is to generate a single, cohesive summary of the provided conversation turns between {speaker_a} and {speaker_b} that highlights the most important insights. The input contains timestamps, speaker identities, and dialogue.

**Requirements:**
 Focus on the key points from both inputs and ensure the final summary combines the insights from both pieces of information.
 You must incorporate specific timestamps into the summary to show when key topics were discussed or when decisions were made (e.g., "At June 1st, 2022, Speaker A introduced the topic of...").

**Input Text:**
{segment_text}

**Note:**
Provide a single, cohesive summary of the conversation, adhering to the requirements above.

**Summary:**
"""

RESPONSE_PROMPT = """
### Role
You are an conversation expert with access to the chat history between **{speaker_a}** and **{speaker_b}** and some relevant information.

### Task
Your goal is to answer the specific **Query** based *only* on the provided **Retrieved Context** (historical relevant information) and **Recent Dialogue** (current context).

### Input Data

**Recent Dialogue (The most recent conversation turns):**
{history}

**Retrieved Context (Relevant historical information with timestamps):**
{retrieved}

**Query:**
{query}

### Instructions
1. Examine all memories that contain information related to the question and synthesize findings from multiple memories if a single entry is insufficient.
2. Based on the context information, answer the *Query* directly and concisely. Do not hallucinate information not present in the provided context. 
3. When answering time-related questions, ALWAYS try to resolve relative terms (e.g., "yesterday," "next Friday," "in two hours") into specific dates and times (e.g., "October 24th," "2023-11-05 at 14:00") based on the provided memory timestamps if possible.

### Answer:
"""

RESPONSE_PROMPT_MERGE = """
    <MEMORY>\n
    The memories linked to the ongoing conversation are:\n
    {retrieved}\n\n
    <QUESTION>\n
    The question is: {query}\n
    When answering questions, be sure to check whether the timestamp of the referenced information matches the timeframe of the question.
    Please respond to the question in English:\n 
"""

RESPONSE_PROMPT_MERGE_2 = """
### Role
You are an conversation expert with access to recent dialogue and some relevant historical information.

### Task
Your goal is to answer the specific **Query** based *only* on the provided **Recent Dialogue** and **Retrieved Context**.

### Input Data

**Recent Dialogue (The most recent conversation turns):**
{history}

**Retrieved Context (Relevant historical information with timestamps):**
{retrieved}

**Query:**
{query}

### Instructions
1. Examine all memories that contain information related to the question and synthesize findings from multiple memories if a single entry is insufficient.
2. Use the recent dialogue as the highest-priority context when it is relevant to the question.
3. Based on the provided context, answer the *Query* directly and concisely. Do not hallucinate information not present in the provided context. 
4. When answering questions, be sure to check whether the timestamp of the referenced information matches the timeframe of the question.

### Answer in English:
"""

RESPONSE_PROMPT_SPLIT = """
    <RECENT DIALOGUE>\n
    The most recent conversation turns are:\n
    {history}\n\n
    <DIALOGUE MEMORY>\n
    The dialogue memories linked to the ongoing conversation are:\n
    {dial_retrieved}\n\n
    <OTHER MEMORY>\n
    The other relevant memories linked to the ongoing conversation are:\n
    {seg_retrieved}\n\n
    <QUESTION>\n
    The question is: {query}\n
    Use the recent dialogue as the highest-priority context when it is relevant to the question.
    When answering questions, be sure to check whether the timestamp of the referenced information matches the timeframe of the question.
    Please respond to the question in English:\n 
"""
