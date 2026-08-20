# HoVer-based reproducibility data

This directory contains data prepared for small-scale CatRAG reproduction and
experimentation:

- `hover.json`: 1,000 HoVer claim instances with their labels, supporting
  facts, questions, answers, and associated contexts.
- `hover_corpus.json`: 9,440 passages prepared as the retrieval corpus for the
  included instances.

## Source and redistribution

The files are derived from the HoVer v1.1 dataset introduced in *HoVer: A
Dataset for Many-Hop Fact Extraction And Claim Verification*. The authoritative
source is the [official HoVer repository](https://github.com/hover-nlp/hover).
The HoVer repository declares the MIT License; its copyright and permission
notice are retained in [`LICENSE-HOVER`](LICENSE-HOVER).

The bundled files were selected and reshaped for CatRAG experiments and are
not a replacement for the complete authoritative HoVer distribution. Obtain
the complete dataset and any later corrections from the official source.

The corpus includes Wikipedia-derived passage text. The HoVer MIT notice does
not relicense that underlying third-party text. Downstream users remain
responsible for complying with applicable Wikipedia attribution and share-alike
terms when redistributing or adapting that text. See
[Wikipedia copyrights](https://en.wikipedia.org/wiki/Wikipedia:Copyrights).

## Citation

Please cite the HoVer paper when using these files:

```bibtex
@inproceedings{jiang2020hover,
  title     = {{HoVer}: A Dataset for Many-Hop Fact Extraction And Claim Verification},
  author    = {Yichen Jiang and Shikha Bordia and Zheng Zhong and Charles Dognin and Maneesh Singh and Mohit Bansal},
  booktitle = {Findings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2020}
}
```

Also cite the CatRAG paper as described in the repository root README when
using this processed form in CatRAG experiments.
