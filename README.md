<h1 align="center"><a href="https://aclanthology.org/2026.findings-acl.290.pdf">Breaking the Static Graph: Context-Aware Traversal for Graph-Based RAG</a></h1>

### CatRAG is a RAG framework built on the HippoRAG 2 architecture that transforms a static knowledge graph into a query-adaptive navigation structure.

This repository provides a reproduced implementation of CatRAG, together with
the prompts and HoVer dataset used in the paper experiments.

<p align="center">
  <img align="center" src="https://github.com/kwunhang/CatRAG/raw/main/images/catrag_method.png" alt="Comparison of HippoRAG 2 and CatRAG graph traversal" />
</p>
<p align="center">
  <b>Comparison of graph traversal between HippoRAG 2 and CatRAG.</b> We illustrate the retrieval process for the multi-hop query “Which university did Marie Curie’s doctoral advisor attend?”. In HippoRAG 2 (top), the static graph structure causes semantic drift; probability mass is diverted to high-weight generic edges (e.g., <i>Marie Curie</i> → <i>Radioactivity</i>), missing the downstream evidence <i>ENS</i>. CatRAG (bottom) prevents this by applying (1) Symbolic Anchoring, injecting “University” as a weak seed, (2) Query-Aware Dynamic Edge Weighting, amplifying relevant paths (e.g., <i>Attend in ENS</i>) while pruning irrelevant ones, and (3) Key-Fact Passage Weight Enhancement, boosting relevant context edge strength. This steers the random walk to successfully retrieve the complete evidence chain for <i>ENS</i>.
</p>

## Latest news

- **2026-08-20:** Released the reproduced CatRAG implementation and dataset.
- **2026-04-14:** The CatRAG paper was accepted to Findings of ACL 2026.

## Paper

- [Breaking the Static Graph: Context-Aware Traversal for Graph-Based RAG](https://aclanthology.org/2026.findings-acl.290.pdf)

## Installation

CatRAG requires Python 3.10 or newer. A virtual environment is recommended.

```bash
git clone https://github.com/kwunhang/CatRAG.git
cd CatRAG
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The base installation supports OpenAI and OpenAI-compatible chat and embedding
endpoints. Install only the optional backends you need:

```bash
python -m pip install -e ".[local]"    # local Transformers/GritLM embeddings
python -m pip install -e ".[bedrock]"  # Amazon Bedrock providers
python -m pip install -e ".[offline]"  # offline OpenIE with vLLM
python -m pip install -e ".[all]"      # all optional backends
```

The `vllm` extra is primarily intended for Linux GPU environments. Model
downloads, provider API calls, and local inference may require substantial
storage, GPU memory, time, or API credits.

## Quick start

Set an OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Then index a small corpus and ask a question:

```python
from catrag import CatRAG

docs = [
    "Marie Curie studied physics and mathematics at the University of Paris.",
    "Marie Curie's doctoral advisor was Gabriel Lippmann.",
    "Gabriel Lippmann attended the École Normale Supérieure.",
]

catrag = CatRAG(
    save_dir="outputs/quickstart",
    llm_model_name="gpt-4o-mini",
    embedding_model_name="text-embedding-3-small",
)

catrag.index(docs)
solutions, _, _ = catrag.rag_qa(
    queries=["Which university did Marie Curie's doctoral advisor attend?"]
)
print(solutions[0].answer)
```

For an OpenAI-compatible local service, pass `llm_base_url` and
`embedding_base_url` to `CatRAG`. Set `OPENAI_API_KEY` to the credential
required by that service. CatRAG supplies a non-secret placeholder only for an
LLM URL containing `localhost` when the variable is absent.

## Repository structure

```text
.
├── src/catrag/          # reproduced CatRAG package
├── reproduce/dataset/  # HoVer-based reproducibility data
├── prompts/             # pointer to packaged prompt implementations
├── images/              # repository figures
├── pyproject.toml       # package metadata and dependencies
├── LICENSE
└── README.md
```

The authoritative runtime prompts live in `src/catrag/prompts` so that the
published prompts and package behavior do not drift apart.

## Acknowledgements

This independently reproduced implementation is built on and adapts parts of
the [HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG) code architecture
and prompt templates.

## Citation

If you find this work useful, please cite the CatRAG paper:

```bibtex
@inproceedings{lau-etal-2026-breaking,
    title = "Breaking the Static Graph: Context-Aware Traversal for Graph-Based {RAG}",
    author = "Lau, Kwun Hang  and
      Zhang, Fangyuan  and
      Ruan, Boyu  and
      Zhou, Yingli  and
      Guo, Qintian  and
      Zhang, Ruiyuan  and
      Zhou, Xiaofang",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.290/",
    doi = "10.18653/v1/2026.findings-acl.290",
    pages = "5849--5863",
    ISBN = "979-8-89176-395-1"
}
```

