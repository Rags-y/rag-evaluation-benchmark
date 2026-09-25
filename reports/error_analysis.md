# Day 6 - Error Analysis

## Overview

This analysis investigates the baseline RAG system's worst failures using the
40-question evaluation dataset and the generated baseline evaluation results.

The purpose is to distinguish retrieval failures from answer-generation
failures and identify the main system bottleneck.

## Evaluation Setup

- Corpus: Python Documentation
- Questions: 40
- Retrieval: dense vector search
- Embedding model: `all-MiniLM-L6-v2`
- Chunk size: 1200 characters
- Chunk overlap: 200 characters
- Top-k: 5
- Abstention threshold: 0.50
- Generator: deterministic local mock generator

The mock generator returns the highest-ranked retrieved context rather than
performing LLM-based answer synthesis. Therefore, generation-related metrics
are diagnostic of the current baseline implementation.

## Failure Categories

### 1. False Abstention

**Example: q015**

Question:

> What does the finally clause do?

The system retrieved the correct document with:

- Retrieval precision: 0.800
- Document recall: 1.000
- Top retrieval score: 0.495
- Abstention: true
- Correctness: 0.000

The relevant `finally` clause content was present in the retrieved contexts,
but the top retrieval score was just below the configured 0.50 abstention
threshold.

This indicates a threshold-related false abstention rather than a retrieval
coverage failure.

### 2. Generation and Synthesis Failures

Several multi-hop questions retrieved the expected source document while
producing low-correctness answers.

Examples include:

- q017: correctness 0.400
- q020: correctness 0.200
- q021: correctness 0.333
- q022: correctness 0.417
- q023: correctness 0.500

Multi-hop document recall was 1.000 overall, while correctness was only 0.476.

This indicates that the baseline generally retrieves relevant evidence but does
not reliably combine or synthesize that evidence into a direct answer.

The primary reason is the deterministic mock generator, which returns a
retrieved context rather than generating an answer from the retrieved
evidence.

### 3. Misleading-Question Failures

Misleading questions expose another generation limitation.

For example, q040 asks which list operation guarantees that every possible
pair of values can be compared successfully before sorting.

The retrieved context explicitly states that some values cannot be compared and
that Python may raise `TypeError`. However, the generated answer returns a
large retrieved fragment rather than directly correcting the false premise.

q040 has:

- Retrieval precision: 0.800
- Document recall: 1.000
- Correctness: 0.312

This demonstrates that relevant retrieval does not guarantee that the system
will recognize and correct a misleading premise.

### 4. Partial Retrieval Coverage

q038 is an example where retrieval coverage is incomplete.

- Retrieval precision: 1.000
- Document recall: 0.500
- Correctness: 0.333

The retrieved contexts contain relevant exception-handling material, but the
system does not retrieve all expected source-document evidence for the
question.

This represents a smaller but measurable retrieval limitation.

### 5. Unanswerable Questions

All eight unanswerable questions were correctly abstained from.

- Questions: q026â€“q033
- Correct abstentions: 8/8
- Abstention rate: 1.000

Their top retrieval scores were substantially below the 0.50 threshold.

This indicates that the current threshold successfully separates the
out-of-corpus questions in this evaluation set, although q015 demonstrates
that the same threshold can also cause false abstention on an answerable
question.

## Top Failure Examples

| Question | Type | Main Failure | Evidence |
|---|---|---|---|
| q015 | Direct factual | False abstention | Relevant context retrieved; score 0.495 |
| q038 | Misleading | Partial retrieval + generation | Recall 0.500; correctness 0.333 |
| q004 | Direct factual | Generation | Recall 1.000; correctness 0.667 |
| q036 | Misleading | Generation | Recall 1.000; correctness 0.455 |
| q040 | Misleading | Generation | Recall 1.000; correctness 0.312 |
| q020 | Multi-hop | Generation/synthesis | Recall 1.000; correctness 0.200 |
| q023 | Multi-hop | Generation/synthesis | Recall 1.000; correctness 0.500 |
| q017 | Multi-hop | Generation/synthesis | Recall 1.000; correctness 0.400 |
| q009 | Direct factual | Generation | Recall 1.000; correctness 0.750 |
| q022 | Multi-hop | Generation/synthesis | Recall 1.000; correctness 0.417 |

## Main Bottleneck

The evidence indicates that **answer generation and synthesis are the primary
bottleneck** in the baseline system.

The retrieval component frequently finds the relevant source document, but
the mock generator returns raw retrieved context instead of producing a
focused answer.

The strongest evidence is the multi-hop category:

- Retrieval precision: 0.900
- Document recall: 1.000
- Correctness: 0.476

Therefore, improving retrieval alone is unlikely to address the largest
observed failure pattern. A stronger generation component that can synthesize
multiple retrieved contexts and explicitly reject misleading premises would
address the observed limitation.

## Secondary Bottleneck: Abstention Threshold

The 0.50 threshold produces a false abstention on q015 at a retrieval score
of 0.495.

At the same time, all eight unanswerable questions correctly abstain.

This demonstrates a threshold trade-off between avoiding unsupported answers
and avoiding false abstentions.

The threshold should therefore be evaluated empirically rather than changed
solely based on one failure.

## Limitations

The generation-related metrics should be interpreted as diagnostic because the
baseline uses a deterministic mock generator.

The faithfulness and answer-relevance metrics are lightweight lexical proxies,
not semantic evaluation models.

Retrieval recall is measured at the document level rather than the individual
chunk level.

## Conclusion

The Day 6 analysis separates retrieval quality from generation quality.

The baseline demonstrates strong document-level retrieval coverage for most
answerable questions, while answer synthesis remains the dominant source of
failure. The analysis also identifies a threshold-related false abstention
and limitations when handling misleading questions.

These findings provide evidence for the next stage of the benchmark without
changing the baseline solely to optimize the current evaluation results.
