# Retrieval Configuration Comparison

## Configurations

| Configuration | Chunk Size | Chunk Overlap | Top-K | Embedding Model |
|---|---:|---:|---:|---|
| Baseline | 1200 | 200 | 5 | all-MiniLM-L6-v2 |
| Small Chunks | 600 | 100 | 5 | all-MiniLM-L6-v2 |

## Results

### Direct Factual

| Metric | Baseline | Small Chunks | Delta |
|---|---:|---:|---:|
| Retrieval Precision | 0.680 | 0.733 | +0.053 |
| Retrieval Recall | 1.000 | 0.933 | -0.067 |
| Answer Relevance | 0.538 | 0.494 | -0.044 |
| Faithfulness | 0.956 | 1.000 | +0.044 |
| Correctness | 0.697 | 0.607 | -0.090 |
| Abstention Rate | 0.067 | 0.000 | -0.067 |

### Multi-Hop

| Metric | Baseline | Small Chunks | Delta |
|---|---:|---:|---:|
| Retrieval Precision | 0.900 | 0.920 | +0.020 |
| Retrieval Recall | 1.000 | 0.950 | -0.050 |
| Answer Relevance | 0.454 | 0.341 | -0.114 |
| Faithfulness | 1.000 | 1.000 | 0.000 |
| Correctness | 0.476 | 0.388 | -0.088 |
| Abstention Rate | 0.000 | 0.000 | 0.000 |

### Unanswerable

| Metric | Baseline | Small Chunks | Delta |
|---|---:|---:|---:|
| Retrieval Precision | 0.000 | 0.000 | 0.000 |
| Retrieval Recall | 0.000 | 0.000 | 0.000 |
| Answer Relevance | 0.000 | 0.000 | 0.000 |
| Faithfulness | 0.542 | 0.472 | -0.069 |
| Correctness | 0.000 | 0.000 | 0.000 |
| Abstention Rate | 1.000 | 1.000 | 0.000 |

### Misleading

| Metric | Baseline | Small Chunks | Delta |
|---|---:|---:|---:|
| Retrieval Precision | 0.886 | 0.914 | +0.029 |
| Retrieval Recall | 0.929 | 1.000 | +0.071 |
| Answer Relevance | 0.391 | 0.312 | -0.080 |
| Faithfulness | 1.000 | 1.000 | 0.000 |
| Correctness | 0.522 | 0.503 | -0.019 |
| Abstention Rate | 0.000 | 0.000 | 0.000 |

## Observations

- The smaller-chunk configuration increased retrieval precision for direct factual, multi-hop, and misleading questions.
- Retrieval recall decreased for direct factual and multi-hop questions with smaller chunks.
- The smaller-chunk configuration achieved complete abstention on the unanswerable questions, matching the baseline.
- Answer relevance and correctness decreased for the smaller-chunk configuration across the answerable categories.
- These answer-generation metrics should be interpreted cautiously because the current benchmark uses a deterministic mock generator that returns the highest-ranked retrieved chunk rather than generating a synthesized answer.

## Interpretation

The results show a measurable trade-off between retrieval precision and retrieval recall when changing chunk size while keeping the embedding model and top-k constant.

The benchmark therefore separates retrieval behavior from generation behavior rather than treating a single answer score as the only evaluation signal.