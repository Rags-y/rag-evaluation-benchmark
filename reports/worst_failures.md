# Worst Failure Analysis

The table below contains the ten highest-scoring failures from the baseline evaluation. Correctly abstained unanswerable questions are excluded from the failure ranking.

| Rank | Question ID | Type | Failure Score | Observed Failure |
|---:|---|---|---:|---|
| 1 | q015 | Direct factual | 3.867 | The system abstained even though the question is answerable from the corpus. |
| 2 | q038 | Misleading | 1.917 | The system returned an irrelevant fragment instead of correcting the false premise. |
| 3 | q004 | Direct factual | 1.848 | Retrieved context was related to lists but did not directly answer what indexing returns. |
| 4 | q036 | Misleading | 1.612 | Retrieved context discussed sorting limitations but the generated response did not directly address the false premise. |
| 5 | q040 | Misleading | 1.476 | Retrieved context was related to list comparison but did not directly answer the misleading claim. |
| 6 | q020 | Multi-hop | 1.411 | Relevant information about `remove()` and `ValueError` was retrieved, but the response was an incomplete document fragment. |
| 7 | q023 | Multi-hop | 1.400 | Relevant information about slicing was retrieved, but the response did not clearly answer the requested operation. |
| 8 | q017 | Multi-hop | 1.360 | Relevant list-method context was retrieved, but the response did not synthesize the combined behavior of `append()` and `pop()`. |
| 9 | q009 | Direct factual | 1.339 | Relevant `remove()` documentation was retrieved, but the response contained surrounding material instead of a concise answer. |
| 10 | q022 | Multi-hop | 1.233 | The retrieved context was related to lists but did not directly answer the stack behavior question. |

## Failure Patterns

### 1. Retrieval and generation are coupled

The baseline generator returns the highest-ranked retrieved chunk rather than synthesizing an answer from the retrieved context. Consequently, a relevant retrieval can still produce a poor final answer.

### 2. Multi-hop questions are difficult

Several multi-hop questions retrieved relevant documentation but failed to combine the required facts into a direct answer.

### 3. Misleading questions expose answer-generation limitations

Questions containing false premises require the system to identify and correct the premise. The current deterministic generator does not perform this reasoning step.

### 4. False abstention

q015 demonstrates that the fixed abstention threshold can reject an answerable question when its highest retrieval score falls below the threshold.

## Main Bottleneck

The failure cases indicate that the current benchmark's largest limitation is the generation stage. Retrieval often returns related documentation, but the mock generator does not synthesize an answer from the retrieved context.

The results should therefore be interpreted primarily as a retrieval benchmark at this stage, while generation-quality metrics provide diagnostic information about the limitations of the deterministic mock generator.