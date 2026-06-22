# Zero-Shot RAG Evaluation: Benchmarking Dense, Approximate, and Hybrid Search on BEIR SciFact

This repository houses a production-grade, empirical benchmarking framework evaluating five information retrieval (IR) strategies across six state-of-the-art dense embedding models. Engineered specifically for optimizing Retrieval-Augmented Generation (RAG) systems, this pipeline explicitly analyzes the mathematical tradeoffs between search accuracy (precision, recall, positional ranking) and computational hardware costs (raw inference overhead vs. real-time query latencies).

---

## 🔬 Project Context: Why BEIR SciFact?

The **BEIR (Benchmarking Information Retrieval)** suite is the global standard for evaluating how well search engines and AI models perform *outside* of their training data (known as zero-shot evaluation). 

This framework utilizes the **SciFact** dataset, which consists of **300 expert-written scientific claims** that must be verified against a corpus of **5,183 scientific abstracts from PubMed**. Because medical literature is dense with highly specialized terminology, it serves as the ultimate stress test for vector embedding spaces and keyword-matching engines alike.

In a production RAG pipeline, if your retriever fails to surface the exact ground-truth document within its top results, the downstream Large Language Model (LLM) will confidently hallucinate an incorrect response. This project benchmarks four algorithmic approaches to maximize retrieval accuracy, minimize latency, and optimize context quality at Rank Cutoff 10 (`@10`).

---

## 📊 Evaluation Results Summary


The master framework concurrently tracks **25 unique system pipelines** (1 standalone lexical baseline + 4 functional search strategies across 6 core neural network architectures), mapping performance metrics side-by-side with localized runtime analytics.


| Search Approach | Total True Hits | Precision@10 | Recall@10 | MRR@10 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ANN Flat Cluster (`IVFFlat`)** | 208 | 0.0693 | 0.6267 | 0.4677 | 0.5059 |
| **Brute Force (`Exact Cosine`)** | 220 | 0.0733 | 0.6667 | 0.5009 | 0.5405 |
| **HNSW Graph (`IndexHNSWFlat`)** | 221 | 0.0737 | 0.6700 | 0.5013 | 0.5415 |
| **BM25 Lexical Only** | 229 | 0.0763 | 0.7033 | 0.5242 | 0.5678 |
| **Hybrid (HNSW + BM25)** | **238** | **0.0793** | **0.7300** | **0.5754** | **0.6122** |


### 📈 Core Engineering Insights
1. **The Domain-Specific Vector Deficit:** Standalone keyword matching (`BM25`) outperformed pure semantic vector spaces (`Brute Force`/`HNSW`) across all metrics. This highlights a classic vector deficit where general-purpose dense models experience vocabulary drift when handling exact medical terms, genes, and chemical mutations.
2. **The Power of Multi-Engine Fusion:** While BM25 proved strong, combining it with structural semantic vectors via the **Hybrid** track generated our peak performance. It reclaimed obscured context documents, driving **Recall@10** to **72%** and **NDCG@10** to an exceptional **0.6122**.
3. **Flawless HNSW Graph Navigation:** The native `IndexHNSWFlat` configuration successfully mirrored (and slightly optimized via floating-point tie-breaking) the retrieval accuracy of Brute Force while fundamentally reducing production search times to sub-millisecond ranges.

---

## 🛠️ System Architecture & Search Tracks

The project executes and logs four parallel retrieval pipelines over the corpus:

                  ┌──► BM25 Lexical Only (Raw Term Frequency Inverse Document Frequency Baseline)
                  ├──► Brute Force (PyTorch Matrix Multiplication with L2 Normalization)
                  ├──► ANN Flat Cluster (FAISS IndexIVFFlat Quantizer; nlist=64, nprobe=16)
[User Query] ─────┼──► HNSW Graph (FAISS IndexHNSWFlat Hierarchical Network Graph)
                  └──► Hybrid Track ──► [HNSW + BM25Okapi] ──► Reciprocal Rank Fusion (RRF)


1. **BM25 Lexical Only:** Evaluates exact term frequencies matching using the `BM25Okapi` sparse tokens matrix on CPU.
2. **Brute Force (Exact Cosine):** Runs exact matrix tensor multiplications utilizing native `torch.mm` across Normalized Document Vectors on GPU/CPU. Serves as our control baseline.
3. **ANN Flat Cluster (`IndexIVFFlat`):** Partitioning index that groups the normalized vector space into 64 distinct Voronoi cells. It accelerates lookups by restricting the search path to the nearest 16 clusters (`nprobe=16`).
4. **HNSW Graph (`IndexHNSWFlat`):** Constructs a multi-layer hierarchical network graph (`M=32`, `efSearch=64`, `efConstruction=64`) for accelerated vector routing, yielding optimal approximate nearest neighbors.
5. **Hybrid (HNSW + BM25):** Executes a concurrent lexical-semantic pipeline. Exact keyword matching from `BM25Okapi` is mathematically merged with structural graphs from HNSW using **Reciprocal Rank Fusion (RRF)**:
   $$RRF\_Score(d \in D) = \frac{1}{k_{rrf} + \text{rank}_{HNSW}(d)} + \frac{1}{k_{rrf} + \text{rank}_{BM25}(d)}$$

---

## 📝 Metric Definitions for RAG Ingestion

We truncate all metrics strictly to the Top 10 results (`@10`) because it matches the token limits, API budgets, and practical context constraints of modern LLM prompt engineering.

* **Recall@10 (Quantity Metric):** Measures if the ground-truth document was successfully caught anywhere inside the Top 10 window. Critical for preventing **LLM Hallucinations**.
* **Precision@10 (Signal-to-Noise Metric):** Measures the proportion of helpful vs. junk documents entering the prompt. High precision keeps prompts clean and limits token billing.
* **MRR@10 & NDCG@10 (Sorting Quality Metrics):** Tracks positional accuracy. LLMs suffer from "lost-in-the-middle" bias and pay the closest attention to information at the absolute top of their prompt window. High MRR and NDCG verify that the strongest evidence is consistently delivered at **Rank 1 or Rank 2**.


## 💾 Optimization & Compatibility Layer

* **Embedding Cache Layer:** Bypasses repetitive encoding iterations by utilizing a local `.npy` vector cache system (`./results/faiss_cache`). If text documents have already been mapped into vector space, they are reloaded into RAM instantly.
* **NumPy 2.0+ Migration Fix:** The statistical evaluation loop is fully decoupled from deprecated legacy code. The metric math engines replace the removed `np.asfarray` functions with explicit type-safe `np.asarray(r, dtype=float)` interfaces.
* **Unified DataFrame Packaging:** Implements type-safe serialization protection via `get_predictions_dataframe` to seamlessly handle discrepancies between string keys (`BM25`/`BEIR`) and localized FAISS integer indexing arrays.

---

## 📂 Project Directory Structure

```text
├── datasets/
│   └── scifact/                    # Contains queries.jsonl, corpus.jsonl, and qrels
├── evaluation/
│   └── metrics.py                  # Module housing get_predictions_dataframe logic
├── results/
│   ├── faiss_cache/                # Cached scifact_embeddings.npy document storage
│   ├── raw_search_predictions.xlsx # Consolidated 15,000-row 5-track prediction file
│   └── final_metric_scorecard.xlsx # Complete evaluation summary dashboard
├── retrievers/
│   └── dense.py                    # Vector Index wrappers (FAISS), BM25 Okapi, and RRF Core
├── app.py                          # Master runtime pipeline orchestration script
└── evaluate_metrics.py             # NumPy 2.0 compatible metric evaluation engine


🚀 Execution Guide
1. Installation
Set up your local virtual environment and install the verified packages:


```
python -m venv rag_env
source rag_env/Scripts/activate     # On Windows use: rag_env\Scripts\activate
pip install pandas numpy rank-bm25 faiss-cpu torch beir openpyxl tqdm
```

2. Step 1: Run the Retrieval Matrix
Execute the main application to load the corpus, build your vector spaces, execute the 4-track searches, and write predictions to an Excel spreadsheet:

```
python app.py
```

3. Step 2: Extract Performance Scorecards
Run the mathematical validation script to evaluate sorting quality, retrieve counts, and extract performance metrics:

```
python evaluate_metrics.py
```