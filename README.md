# RAG Evaluation Benchmark

A reproducible benchmark for evaluating Retrieval-Augmented Generation (RAG) systems. The project evaluates retrieval quality, answer quality, abstention behavior, and failure modes using a public Python documentation corpus.

The benchmark separates retrieval performance from answer-generation performance and compares two retrieval configurations on the same 40-question evaluation dataset.

## Objectives

The benchmark evaluates:

* Retrieval precision
* Document-level retrieval recall
* Answer relevance
* Answer faithfulness
* Answer correctness
* Exact match
* Abstention behavior
* Retrieval and generation failure modes
* Retrieval configuration trade-offs

## Corpus

The benchmark uses publicly available documentation from the official Python documentation website.

Source:

https://docs.python.org/3/

The evaluation corpus contains six documentation pages:

1. Python Introduction
2. Control Flow Tools
3. Data Structures
4. Classes
5. Input and Output
6. Errors and Exceptions

The corpus is downloaded automatically and cached locally.

## Evaluation Dataset

The benchmark contains **40 evaluation questions** divided into four categories:

| Category       | Questions | Purpose                                            |
| -------------- | --------: | -------------------------------------------------- |
| Direct factual |        15 | Test straightforward retrieval and answering       |
| Multi-hop      |        10 | Test combining information from retrieved evidence |
| Unanswerable   |         8 | Test abstention for information outside the corpus |
| Misleading     |         7 | Test handling of false or misleading premises      |

The dataset is stored at:

```text
data/evaluation/questions.json
```

Each question includes:

* Question ID
* Question text
* Question type
* Answerability
* Expected answer
* Expected source document IDs

## Retrieval Configurations

Two dense retrieval configurations are evaluated using the same embedding model.

### Baseline

```yaml
Embedding model: all-MiniLM-L6-v2
Chunk size: 1200
Chunk overlap: 200
Top-k: 5
Abstention threshold: 0.50
```

Configuration:

```text
configs/default.yaml
```

### Small-Chunk Variant

```yaml
Embedding model: all-MiniLM-L6-v2
Chunk size: 600
Chunk overlap: 100
Top-k: 5
Abstention threshold: 0.50
```

Configuration:

```text
configs/variant_small_chunks.yaml
```

The embedding model and top-k remain constant so that the comparison focuses on the effect of chunk size and overlap.

## Evaluation Metrics

### Retrieval

**Retrieval Precision**

Measures the proportion of retrieved contexts that belong to the expected source documents.

**Document Recall**

Measures whether the expected source documents were retrieved.

### Answer Quality

**Answer Relevance**

Measures lexical overlap between the question and generated answer.

**Faithfulness**

Measures lexical support for answer content in the retrieved contexts.

**Correctness**

Measures token overlap between the generated answer and expected answer.

**Exact Match**

Checks whether the normalized generated answer exactly matches the expected answer.

### Abstention

The benchmark measures whether the system abstains when retrieved evidence is insufficient.

Unanswerable questions are expected to produce an abstention rather than an unsupported answer.

## Architecture

```text
                    Evaluation Dataset
                           |
                           v
                     User Question
                           |
                           v
                  Dense Vector Retrieval
                           |
                           v
                    Top-k Contexts
                           |
                           v
                    RAG Generation
                           |
                           v
                     Final Answer
                           |
                           v
                 Evaluation Metrics
                           |
              +------------+------------+
              |            |            |
         Retrieval     Answer       Abstention
          Metrics      Metrics        Metrics
```

The current baseline uses a deterministic local mock generator for reproducible evaluation. It returns the highest-ranked retrieved context rather than performing LLM-based answer synthesis.

This makes the generation-related results useful for diagnosing the current pipeline, while avoiding dependence on an external API.

## Project Structure

```text
rag-evaluation-benchmark/
|
├── configs/
│   ├── default.yaml
│   └── variant_small_chunks.yaml
|
├── data/
│   ├── evaluation/
│   │   └── questions.json
│   ├── raw/
│   ├── processed/
│   ├── processed_small/
│   ├── index/
│   └── index_small/
|
├── plots/
│   ├── retrieval_precision_recall.png
│   ├── answer_quality_comparison.png
│   └── abstention_rate_comparison.png
|
├── reports/
│   ├── error_analysis.md
│   ├── retrieval_comparison.md
│   └── worst_failures.md
|
├── results/
│   ├── baseline_results.json
│   ├── baseline_summary.json
│   ├── variant_small_chunks_results.json
│   ├── variant_small_chunks_summary.json
│   ├── retrieval_comparison.json
│   └── worst_failures.json
|
├── src/
│   ├── build_corpus.py
│   ├── build_index.py
│   ├── retrieval_demo.py
│   └── rag_eval/
│       ├── chunker.py
│       ├── loader.py
│       ├── schemas.py
│       ├── vector_store.py
│       ├── rag_pipeline.py
│       ├── evaluation_dataset.py
│       ├── evaluation_metrics.py
│       ├── evaluation_runner.py
│       ├── evaluation_summary.py
│       ├── retrieval_comparison.py
│       ├── error_analysis.py
│       ├── plot_results.py
│       ├── plot_answer_quality.py
│       └── plot_abstention.py
|
├── tests/
├── compare_retrieval.py
├── run_evaluation.py
├── run_rag.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install the project in editable mode:

```powershell
python -m pip install -e .
```

## Build the Corpus

The corpus can be downloaded and processed using:

```powershell
python src/build_corpus.py
```

This creates the processed corpus used by the retrieval pipeline.

## Build the Vector Index

The baseline vector index uses the default configuration:

```powershell
python src/build_index.py
```

The small-chunk configuration can be built using its configuration:

```powershell
python src/build_index.py --config configs/variant_small_chunks.yaml
```

Generated corpus and index files are excluded from Git because they can be recreated locally.

## Run the Baseline RAG

A single query can be tested with:

```powershell
python run_rag.py --query "What are Python lists?"
```

## Run the Evaluation

After setup and index generation, the complete benchmark can be executed with:

```powershell
python run_evaluation.py --config configs/default.yaml
```

This is the primary evaluation command.

Results are written to:

```text
results/baseline_results.json
results/baseline_summary.json
```

The alternative configuration can be evaluated using:

```powershell
python run_evaluation.py --config configs/variant_small_chunks.yaml --index-dir data/index_small --output-name variant_small_chunks
```

## Compare Retrieval Configurations

After both configurations have been evaluated:

```powershell
python compare_retrieval.py
```

The comparison is saved to:

```text
results/retrieval_comparison.json
```

## Generate Plots

The benchmark includes three plots:

```powershell
python -m rag_eval.plot_results
```

```powershell
python -m rag_eval.plot_answer_quality
```

```powershell
python -m rag_eval.plot_abstention
```

Generated plots:

```text
plots/retrieval_precision_recall.png
plots/answer_quality_comparison.png
plots/abstention_rate_comparison.png
```

## Run Error Analysis

The worst failures can be analyzed with:

```powershell
python -m rag_eval.error_analysis
```

The analysis produces the top 10 failure cases and helps distinguish retrieval failures from answer-generation failures.

Detailed analysis is available in:

```text
reports/error_analysis.md
reports/worst_failures.md
```

## Testing

Run the complete test suite with:

```powershell
pytest -q
```

The current project contains **37 automated tests** covering chunking, retrieval, vector storage, dataset validation, evaluation metrics, evaluation execution, summaries, and retrieval comparison.

## Results Summary

The baseline evaluation produced the following category-level results:

| Question Type  | Retrieval Precision | Document Recall | Correctness | Abstention Rate |
| -------------- | ------------------: | --------------: | ----------: | --------------: |
| Direct factual |               0.680 |           1.000 |       0.697 |           0.067 |
| Multi-hop      |               0.900 |           1.000 |       0.476 |           0.000 |
| Unanswerable   |               0.000 |           0.000 |       0.000 |           1.000 |
| Misleading     |               0.886 |           0.929 |       0.522 |           0.000 |

The small-chunk configuration produced:

| Question Type  | Retrieval Precision | Document Recall | Correctness | Abstention Rate |
| -------------- | ------------------: | --------------: | ----------: | --------------: |
| Direct factual |               0.733 |           0.933 |       0.607 |           0.000 |
| Multi-hop      |               0.920 |           0.950 |       0.388 |           0.000 |
| Unanswerable   |               0.000 |           0.000 |       0.000 |           1.000 |
| Misleading     |               0.914 |           1.000 |       0.503 |           0.000 |

The comparison demonstrates a measurable trade-off between retrieval precision and recall when changing chunk size.

## Error Analysis Findings

The Day 6 analysis identified several important failure patterns.

### Generation and Synthesis

The strongest observed limitation is answer generation.

The baseline often retrieves the expected source document but returns a raw retrieved context rather than synthesizing a focused answer.

This is particularly visible in multi-hop questions, where document-level recall reached 1.000 while correctness remained 0.476.

### False Abstention

Question q015 was answerable and relevant evidence was retrieved, but its top retrieval score was 0.495, just below the 0.50 abstention threshold.

This resulted in a false abstention.

### Unanswerable Questions

All eight unanswerable questions were correctly abstained from.

```text
Correct abstentions: 8/8
```

### Misleading Questions

Misleading questions showed that retrieving relevant evidence does not automatically cause the system to reject a false premise or formulate the appropriate correction.

## Limitations

The current benchmark has several limitations:

* The generator is a deterministic mock generator rather than a production LLM.
* Answer relevance and faithfulness use lightweight lexical metrics.
* Correctness uses token-overlap rather than semantic grading.
* Retrieval recall is measured at the document level.
* The evaluation corpus is intentionally small.
* The abstention threshold is fixed at 0.50 rather than optimized automatically.

These limitations are documented so that the benchmark results are interpreted as an evaluation of the current baseline rather than a production-quality RAG system.

## Reproducibility

The benchmark uses:

* A fixed evaluation dataset
* Fixed retrieval configurations
* A fixed embedding model
* A deterministic local generator
* Configuration files for experiment settings
* Automated tests
* Machine-readable JSON outputs
* Markdown reports
* Generated plots

Generated corpus, indexes, results, plots, and other runtime artifacts can be recreated locally and are excluded from the normal Git workflow where appropriate.

## Deliverables

The completed benchmark contains:

* Public evaluation corpus
* 40-question evaluation dataset
* Baseline vector-search RAG
* Two retrieval configurations
* Configurable evaluation pipeline
* Machine-readable JSON results
* Retrieval comparison
* Three evaluation plots
* Top 10 worst-failure analysis
* Error-analysis report
* Automated test suite
* Reproducible README and setup instructions

## Author
Raghav Kumar
