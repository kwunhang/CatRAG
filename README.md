<h1 align="center"> CatRAG: Context-Aware Transversal for Robust RAG </h1>

### CatRAG is a RAG framework builds on the HippoRAG 2 architecture and transforms the static KG into query-adaptive navigation structure.

This repository provides the **prompts** and **Hover dataset** for the paper *Breaking the Static Graph: Context-Aware Traversal for Robust Retrieval-Augmented Generation*.

Code will be released in a later update.

<p align="center">
  <img align="center" src="https://github.com/kwunhang/CatRAG/raw/main/images/catrag_method.png" />
</p>
<p align="center">
  <b>Comparison of graph traversal between HippoRAG 2 and CatRAG.</b> We illustrate the retrieval process for the multi-hop query “Which university did Marie Curie’s doctoral advisor attend?”. In HippoRAG 2 (top), the static graph structure causes semantic drift; probability mass is diverted to high-weight generic edges (e.g., *Marie Curie* → *Radioactivity*), missing the downstream evidence *ENS*. CatRAG (bottom) prevents this by applying (1) Symbolic Anchoring, injecting “University” as a weak seed, (2) Query-Aware Dynamic Edge Weighting, amplifying relevant paths (e.g., *Attend in ENS*) while pruning irrelevant ones, and (3) Key-Fact Passage Weight Enhancement, boosting relevant context edge strength. This steers the random walk to successfully retrieve the complete evidence chain for *ENS*.
</p>


## What is included

- Prompts used in the paper experiments (see `prompts/`).
- Dataset / evaluation instances used in the paper (see `reproduce/`).

## Repository structure

- `prompts/`  
  Prompt templates and/or prompt sets used for evaluation.

- `reproduce/`  
  Evaluation dataset files (queries, documents/corpus references, and ground truth where applicable).

- `images/`  
  Images used in this repository.

- `README.md`  
  This document.

## TODO:

- [ ] Add CatRAG logic code 

Please feel free to open an issue or PR if you have any questions or suggestions.
