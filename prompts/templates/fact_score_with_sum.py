system_prompt_optimized = """You are a knowledge graph reasoning expert. Score neighbor entities (0-10) on their utility for answering a QUERY.

### Input Data
1. A user QUERY.
2. The CURRENT ENTITY node we are exploring.
3. A set of RETRIEVED FACTS (trusted evidence).
4. A list of NEIGHBORS, each with:
   - The specific LINKING TRIPLET(s) connecting the current entity to this neighbor.
   - A short summary of the neighbor information.

### Scoring Criteria
- **10 (Solution):** The neighbor IS the answer or contains it.
- **7-9 (Bridge):** Critical step in the reasoning chain (e.g., Subject -> Attribute).
- **4-6 (Weak):** Valid semantic link, but tangential to query intent.
- **0-3 (Noise):** Irrelevant, generic, or contradicts facts.

### Rules
1. **Trust Facts:** If a neighbor contradicts RETRIEVED FACTS, score 0.
2. **Output Format:** - `ID (Entity Name): Score` (if Score < 4)
   - `ID (Entity Name): Score | Concise reasoning` (if Score >= 4)
3. **Constraint:** You must copy the Entity Name exactly as it appears in the input.
"""

example1 = """QUERY: "Time Out's 100 best British films was topped by a film adapted from a short story by which author?"
RETRIEVED FACTS: ("Don't Look Now", "topped", "Time Out's 100 best British films")
CURRENT ENTITY NODE: "Don't Look Now"

NEIGHBORS:
[1] "Daphne du Maurier" | LINKING FACT: ("Daphne du Maurier", "write", "Don't Look Now") | SUMMARY: British novelist and short story writer known for works like "Rebecca" and "The Birds". Many of her stories have been adapted into films, including "Don't Look Now" which was based on her short story.
[2] "Nicolas Roeg" | LINKING FACT: ("Don't Look Now", "directed by", "Nicolas Roeg") | SUMMARY: Nicolas Roeg was an English film director. He is best known for directing the 1973 British horror film 'Don't Look Now', which is an adaptation of a short story by Daphne du Maurier.
[3] "Julie Christie" | LINKING FACT:  ("Julie Christie", "stars", "Don't Look Now") | SUMMARY: Julie Christie starred in Don't Look Now. Julie Christie is British actress.
[4] "British film" | LINKING FACT: ("Don't Look Now", "instance of", "British film") | SUMMARY: There is no single name for British films; they are simply called British films or films made in the UK.
[5] "British film industry" | LINKING FACT: ("Don't Look Now", "produced within", "British film industry") | SUMMARY: Broader category connecting films, directors, awards, and publications. Has relationships with ranking entities, publication entities, and institutional databases.
[6] "Short story adaptations" | LINKING FACT: ("Don't Look Now", "is", "short story adaptation") | SUMMARY: Conceptual entity linking literary works to their film adaptations. Connects authors, stories, films, and adaptation credits.

Output ONE line per neighbor: `ID (Entity Name): Score | (Reasoning if Score >= 4)`
"""


example1_output = """Answer:
1 (Daphne du Maurier): 10 | The link directly identifies the author whose short story was adapted. This neighbor is the answer.
2 (Nicolas Roeg): 4 | The link connects to the director. This may lead indirectly to production details but does not target the short story author.
3 (Julie Christie): 2
4 (British film): 2
5 (British film industry): 1
6 (Short story adaptations): 7 | The link bridges from the film to the adaptation concept and then to underlying short stories and their authors, making it a strong but indirect reasoning step toward the answer.
"""

example2 = """QUERY: "Which treaty formally ended the world war sparked by the assassination of the heir to the Austro-Hungarian throne in Sarajevo?"
RETRIEVED FACTS: ("archduke franz ferdinand of austria", "was assassinated in", "sarajevo")
CURRENT ENTITY NODE: archduke franz ferdinand of austria

NEIGHBORS:
[1] "assassination of archduke franz ferdinand" | LINKING FACT: ("archduke franz ferdinand of austria", "victim of", "assassination of archduke franz ferdinand") | SUMMARY: The June 28, 1914 assassination in Sarajevo that triggered the July Crisis and served as the immediate catalyst for World War I
[2] "world war i" | LINKING FACT: ("archduke franz ferdinand of austria", "sparked", "world war i") | SUMMARY: Global war from 1914 to 1918, also known as the Great War, involving many world powers
[3] "austro hungarian empire" | LINKING FACT: ("archduke franz ferdinand of austria", "heir to the throne of", "austro hungarian empire") | SUMMARY: Dual monarchy in Central Europe from 1867 to 1918, ruled by the Habsburg dynasty
[4] "sarajevo" | LINKING FACT: ("archduke franz ferdinand of austria", "was assassinated in", "sarajevo") | SUMMARY: Capital city of Bosnia and Herzegovina, site of the 1914 assassination that sparked World War I

Output ONE line per neighbor: `ID (Entity Name): Score | (Reasoning if Score >= 4)`
"""

example2_output = """Answer:
1 (assassination of archduke franz ferdinand): 6 | While semantically relevant, we already know from retrieved context that he was assassinated. Since we have a direct link to "World War I", visiting this event node is a redundant step for finding the treaty.
2 (world war i): 9 | The link directly connects the assassination to the specific war whose ending treaty the query asks about, making this a crucial bridge toward the treaty entity.
3 (austro hungarian empire): 5 | This neighbor entity helps confirm the heir relation of current entity,  but it does not lead to the specific ending treaty.
4 (sarajevo): 3
"""


prompt_template = [
    {"role": "system", "content": system_prompt_optimized},
    {"role": "user", "content": example1},
    {"role": "assistant", "content": example1_output},
    {"role": "user", "content": example2},
    {"role": "assistant", "content": example2_output},
    {"role": "user", "content": 
        """QUERY: ${query}
RETRIEVED FACTS: ${seed_triplet}
CURRENT ENTITY NODE: ${seed_entity}

NEIGHBORS:
${neighbor_entities}
Output ONE line per neighbor: `ID (Entity Name): Score | (Reasoning if Score >= 4)`
"""}
]

