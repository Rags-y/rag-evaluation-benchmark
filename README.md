# RAG Evaluation Benchmark

A small, reproducible benchmark for evaluating Retrieval-Augmented Generation (RAG) systems.

The project separates retrieval quality from answer quality and compares multiple retrieval configurations on the same evaluation dataset.

## Goals

The benchmark evaluates:

- Retrieval relevance
- Answer faithfulness
- Answer relevance
- Answer correctness
- Abstention quality
- Retrieval and generation failure modes

## Corpus

The initial corpus is a small collection of publicly available Python documentation pages from the official Python documentation.

Source:

https://docs.python.org/3/

The corpus is downloaded automatically and cached locally.

## Project Structure

```text
rag-evaluation-benchmark/
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── plots/
├── reports/
├── results/
│
├── src/
│   └── rag_eval/
│       ├── chunker.py
│       ├── loader.py
│       └── schemas.py
│
├── tests/
│
├── requirements.txt
└── README.md