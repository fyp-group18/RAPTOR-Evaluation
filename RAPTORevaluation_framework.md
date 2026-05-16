# Evaluation Framework

## Overview

We validate our RAPTOR-based retrieval-augmented generation (RAG) system for aircraft maintenance documentation through a three-benchmark evaluation strategy. Each benchmark targets a distinct capability required for production deployment in the aviation maintenance domain. Together, they provide coverage of the system's retrieval quality, multi-page reasoning, and long-document hierarchical comprehension.

| Benchmark | Capability Tested | Metric | Published SOTA |
|---|---|---|---|
| MP-DocVQA | Multi-page document retrieval + QA | ANLS | 73.9% (GRAM) |
| QASPER | Long-document comprehension via hierarchical retrieval | F1 | 55.7% (RAPTOR + GPT-4) |
| DocVQA | Single-page document visual QA | ANLS | — |

## Benchmark Selection Rationale

### MP-DocVQA — Multi-Page Document Visual Question Answering

Aircraft maintenance manuals are inherently multi-page documents, often spanning hundreds of pages per chapter. MP-DocVQA (Tito et al., 2023) evaluates a system's ability to retrieve the correct page from a multi-page document and generate an accurate answer. It contains 927 documents with approximately 5,000 questions, making it the most established benchmark for multi-page document QA.

**Why we compare against it:** MP-DocVQA directly measures the core retrieval challenge our system faces — given a query about a specific procedure or specification, can the system identify the correct page(s) from a lengthy document and extract the answer? This is the primary interaction mode for maintenance technicians using our system.

**Published baselines we compare against:**

| Method | Params (M) | ANLS (%) | Notes |
|---|---|---|---|
| GRAM (Blau et al., 2024) | 281 | 73.9 | Current SOTA; graph-based multi-page reasoning |
| RAG-Qwen2.5-VL (López et al., 2025) | 7,601 | 73.7 | RAG with 7B VLM + retrieval stack |
| RAG-VT5 (López et al., 2025) | 824 | 63.1 | RAG with lightweight textual model |
| Hi-VT5 (Tito et al., 2023) | 316 | 61.8 | Hierarchical attention baseline |
| Pix2Struct-SA-retrieval | 273 | 62.0 | Self-attention retrieval variant |

### QASPER — Question Answering over Scientific Papers

QASPER (Dasigi et al., 2021) contains 5,049 questions across 1,585 NLP papers, requiring retrieval and synthesis of information distributed across full-length documents. Answer types span extractive, abstractive, yes/no, and unanswerable categories.

**Why we compare against it:** QASPER tests the value of RAPTOR's hierarchical tree structure — whether higher-level summary nodes improve retrieval for questions that require synthesising information across multiple sections. Aircraft maintenance queries often require cross-referencing between procedural steps, system descriptions, and parts lists scattered across a document. QASPER is the benchmark where RAPTOR demonstrated its clearest advantage over flat retrieval baselines.

**Published baselines we compare against:**

| Method | F1 (%) | Notes |
|---|---|---|
| RAPTOR + GPT-4 (Sarthi et al., 2024) | 55.7 | SOTA; tree-structured retrieval |
| CoLT5 XL (Ainslie et al., 2023) | 53.9 | Conditional long-context transformer |
| LongT5 XL (Guo et al., 2022) | 53.1 | Long-range transformer |
| DPR + GPT-4 | 53.0 | Dense passage retrieval baseline |
| BM25 + GPT-4 | 50.2 | Sparse retrieval baseline |
| RAPTOR + GPT-3 | 53.1 | Tree retrieval with weaker LLM |

### DocVQA — Document Visual Question Answering

DocVQA evaluates single-page document understanding with visual elements — forms, invoices, and reports with mixed text and layout. It complements MP-DocVQA by isolating per-page comprehension quality from multi-page retrieval.

## Evaluation Design Summary

The three benchmarks form a complementary evaluation matrix:

| Dimension | MP-DocVQA | QASPER | DocVQA |
|---|---|---|---|
| Multi-page retrieval | ✓ | — | — |
| Hierarchical synthesis | — | ✓ | — |
| Visual document comprehension | ✓ | — | ✓ |
| Public comparability | ✓ | ✓ | ✓ |

This design ensures that strong results cannot be attributed to a single favourable benchmark. MP-DocVQA validates core retrieval over multi-page documents. QASPER validates the contribution of RAPTOR's tree structure for cross-section synthesis. DocVQA validates single-page visual comprehension. A system that performs well on all three has demonstrated general retrieval and comprehension capability.
