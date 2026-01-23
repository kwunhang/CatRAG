# node_summarize_system = """You are a precise factual relevance assessor. Your task is to evaluate the relevance of individual factual triplets for answering a specific user query within a potential multi-step reasoning chain. You must output only a list of scores with brief explanations based on the strict scoring rubric.
# """


query_prompt_one_shot_input = """--- Task ---
Generate a concise, entity-focused summary that captures the core identity and key relationships of a given entity based on its associated fact triplets.

--- Instructions ---
1. **Input Format**: You will receive:
   - A `target_entity` (the entity being summarized)
   - A `fact_triplets` list in JSON format containing relationships where this entity appears

2. **Output Requirements**:
   - Focus on the **target entity** as the summary's subject
   - Integrate ALL key relationships from the provided triplets
   - Explain **what the entity is** and **what it connects to** through its relationships
   - Maintain strict coherence and factual accuracy
   - Maximum length: ${summary_length} tokens
   - Language: ${language} (preserve proper nouns in original form when needed)

3. **Content Guidelines**:
   - Start with the entity's core identity/type
   - Group related relationships logically (e.g., all professional roles together)
   - Highlight notable connections to other significant entities
   - Avoid listing facts mechanically - synthesize into narrative form

--- Example Structure ---
[Entity Name] is a [core type/description] known for [key attributes]. It [main relationships/activities] with entities such as [notable connections]. Specifically, it [detailed relationship patterns] and [notable characteristics].

--- Input ---
Target node: alex o loughlin
Fact Triplets: {
('alex o loughlin', 'appeared in', 'moonlight')
('alex o loughlin', 'is', 'australian actor and director')
('alex o loughlin', 'was born on', '24 august 1976')
('alex o loughlin', 'appeared in', 'oyster farmer')
('alex o loughlin', 'plays', 'lieutenant commander steve mcgarrett')
('alex o loughlin', 'appeared in', 'the back up plan')
('alex o loughlin', 'works for', 'cbs')
('alex o loughlin', 'starred in', 'three rivers')
('alex o loughlin', 'starred in', 'moonlight')
('alex o loughlin', 'appeared in', 'three rivers')
('alex o loughlin', 'is an', 'australian actor')
('alex o loughlin', 'had starring roles in', 'oyster farmer')
('alex o loughlin', 'is a', 'writer')
('alex o loughlin', 'had starring roles in', 'the back   up plan')
('alex o loughlin', 'is a', 'director')
}
"""

# query_prompt_one_shot_output = """
# Answer: Alex O'Loughlin is an Australian actor, director, and writer, born on 24 August 1976. He is best known for his work with the CBS network, where he plays Lieutenant Commander Steve McGarrett. O'Loughlin has starred in several television series for CBS, including "Moonlight" and "Three Rivers." His film career includes starring roles in movies such as "Oyster Farmer" and "The Back-up Plan," showcasing his work as an actor on both the big and small screens.
# """
query_prompt_one_shot_output = """
Answer: Alex O'Loughlin is an Australian actor, director, and writer, born on 24 August 1976. He works for CBS, where he plays Lieutenant Commander Steve McGarrett, and has starred in series like "Moonlight" and "Three Rivers" and films such as "Oyster Farmer" and "The Back Up Plan".
"""

# query_prompt_template = """
# Question: {}

# """
prompt_template = [
    # {"role": "system", "content": node_summarize_system},
    {"role": "user", "content": query_prompt_one_shot_input},
    {"role": "assistant", "content": query_prompt_one_shot_output},
    {"role": "user", "content": 
        """Target node: ${entity}
Fact Triplets: ${fact_triplets}
"""}
]