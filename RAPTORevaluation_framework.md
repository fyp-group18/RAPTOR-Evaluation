# Evaluation Framework

## Overview

We validate our RAPTOR-based retrieval-augmented generation (RAG) system for aircraft maintenance documentation through a four-benchmark evaluation strategy. Each benchmark targets a distinct capability required for production deployment in the aviation maintenance domain. Together, they provide comprehensive coverage of the system's retrieval quality, multi-page reasoning, structured table comprehension, and domain-specific accuracy.

| Benchmark | Capability Tested | Metric | Published SOTA |
|---|---|---|---|
| MP-DocVQA | Multi-page document retrieval + QA | ANLS | 73.9% (GRAM) |
| QASPER | Long-document comprehension via hierarchical retrieval | F1 | 55.7% (RAPTOR + GPT-4) |
| SSTQA | Semi-structured table parsing + QA | Accuracy / ROUGE-L | 72.39% / 52.19% (ST-Raptor) |
| Custom Aircraft QA | Domain-specific retrieval over maintenance manuals | ANLS / F1 | — (no prior baseline) |

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

### SSTQA — Semi-Structured Table Question Answering

SSTQA (from ST-Raptor, 2025) contains 764 QA pairs over 102 tables curated from 2,031 real-world tables spanning 19 scenarios including administrative and operational management. Tables feature complex nested structures: multi-row/column headers, merged cells, and irregular layouts (average nesting depth 2.52, significantly deeper than WikiTQ at 1.30).

**Why we compare against it:** Aircraft maintenance manuals contain extensive semi-structured tables — parts lists with hierarchical assemblies, fault isolation decision tables, inspection interval matrices, and torque specification tables. These tables exhibit the same structural complexity (nested headers, merged cells, multi-level hierarchies) that SSTQA is designed to evaluate. Unlike TAT-DQA, which focuses on financial numerical reasoning, SSTQA tests structural comprehension of complex table layouts, which is the primary challenge for our domain. It is also domain-agnostic, avoiding the confound of finance-specific terminology that could disadvantage a general-purpose system.

**Published baselines we compare against:**

| Method | Accuracy (%) | ROUGE-L (%) | Notes |
|---|---|---|---|
| ST-Raptor (2025) | 72.39 | 52.19 | SOTA; hierarchical table decomposition |
| GPT-4o | 66.45 | 43.86 | Foundation model baseline |
| DeepSeek-V3 | 63.22 | 46.17 | Foundation model baseline |
| ReAcTable | 37.24 | — | Agent-based method |
| TAT-LLM | 39.78 | — | Finance-tuned LLM |
| mPLUG-DocOwl1.5 | 29.65 | — | Vision-language model |

### Custom Aircraft Maintenance QA — Domain Validation

No existing public benchmark captures the specific document structures, terminology, and query patterns encountered in aircraft maintenance manuals. We therefore construct a custom evaluation set to validate domain fitness.

**Construction methodology:**
- Source: [N] pages sampled from [specific manual type, e.g., AMM, CMM, IPC, or TSM] covering [aircraft type/system]
- Sampling: stratified by content type — procedural text, fault isolation tables, parts lists with exploded diagrams, system descriptions, and inspection schedules
- Annotation: [N] QA pairs authored by [qualified maintenance engineers / domain SMEs], covering:
  - **Extractive:** direct lookup of specifications, part numbers, torque values
  - **Cross-referencing:** questions requiring information from both text and table on the same or adjacent pages (e.g., "What is the replacement interval for the part listed in Table X that corresponds to system Y?")
  - **Procedural:** multi-step troubleshooting queries requiring retrieval of the correct fault isolation sequence
  - **Numerical reasoning:** questions involving calculations over table values (intervals, tolerances, limits)
- Scale target: 200–400 QA pairs across 50–100 document pages

**Why this is necessary:** Public benchmarks validate general capability but cannot measure whether the system correctly handles ATA chapter structures, effectivity codes, IPC figure/item number references, or MEL dispatch categories. A reviewer or deployment stakeholder would rightly question whether QASPER performance on NLP papers transfers to maintenance manuals. The custom eval closes this gap.

**Baselines:** Since no prior system has been evaluated on this exact dataset, we establish baselines by running:
1. Flat chunking + BM25 retrieval (no RAPTOR tree)
2. Flat chunking + dense retrieval (bi-encoder only, no tree)
3. Our full RAPTOR pipeline

This isolates the contribution of hierarchical retrieval in the target domain.

## Evaluation Design Summary

The four benchmarks form a complementary evaluation matrix:

| Dimension | MP-DocVQA | QASPER | SSTQA | Custom Aircraft |
|---|---|---|---|---|
| Multi-page retrieval | ✓ | — | — | ✓ |
| Hierarchical synthesis | — | ✓ | — | ✓ |
| Table structure parsing | — | — | ✓ | ✓ |
| Domain-specific content | — | — | — | ✓ |
| Public comparability | ✓ | ✓ | ✓ | — |

This design ensures that strong results cannot be attributed to a single favourable benchmark. MP-DocVQA validates core retrieval over multi-page documents. QASPER validates the contribution of RAPTOR's tree structure for cross-section synthesis. SSTQA validates structured table comprehension without relying on a finance-specific dataset. The custom aircraft eval validates real-world deployment readiness. A system that performs well on all four has demonstrated both general capability and domain fitness.
