# Zero-Shot Retriever Evaluation: Benchmarking Dense, Approximate, and Hybrid Search on BEIR SciFact

This repository houses an empirical benchmarking framework evaluating 5 information retrieval (IR) strategies across 6 state-of-the-art dense embedding models. Engineered specifically for optimizing Retrieval-Augmented Generation (RAG) systems, this pipeline explicitly analyzes the mathematical tradeoffs between search accuracy (precision, recall, positional ranking) and computational hardware costs (raw inference overhead vs. real-time query latencies).

---

## 🔬 Project Context: Why BEIR SciFact?

This pipeline uses the SciFact dataset from the BEIR suite to run out-of-domain evaluation.

Corpus: 5,183 scientific abstracts (PubMed)

Queries: 300 expert-written claims


Why this matters for RAG: Specialized medical jargon breaks weak semantic spaces. If a retriever cannot surface relevant documents within the top results, the downstream LLM lacks the context to generate accurate answers. The Author measure 5 distinct search tracks at @10 to find the exact point where retrieval quality meets hardware efficiency.

---

## 📊 Evaluation Results Summary


The master framework concurrently tracks **25 unique system pipelines** (1 standalone lexical baseline + 4 functional search strategies across 6 core neural network architectures), mapping performance metrics side-by-side with localized runtime analytics.

1) BM25 Lexicon
2) Brute Force Search
3) ANN (Approximate Nearest Neighbours) Search
4) HNSW Search
5) Hybrid (HNSW + BM25) Search


## 🎛️ Model Selection Strategy

The author deliberately selected 6 models representing three distinct architectural approaches to dense retrieval. This allows us to map out a clear performance curve across different model sizes, parameter counts, and fine-tuning paradigms.


| Model Family | Model Name on HuggingFace | Dim | Parameters | Architectural Archetype |
| :--- | :--- | :---: | :---: | :--- |
| **Legacy Baseline** | `msmarco-distilbert-base-v4` | 768 | 66M | Cross-Domain Bi-Encoder |
| **BGE Super-Family** | `BAAI/bge-base-en-v1.5` | 768 | 110M | Asymmetric Instruction-Tuned |
| **BGE Super-Family** | `BAAI/bge-large-en-v1.5` | 1024 | 326M | Asymmetric Instruction-Tuned |
| **E5 Super-Family** | `intfloat/e5-base-v2` | 768 | 110M | Prefix-Guided Contrastive Encoder |
| **E5 Super-Family** | `intfloat/e5-large-v2` | 1024 | 326M | Prefix-Guided Contrastive Encoder |
| **Neural Contrastive** | `facebook/contriever-msmarco` | 768 | 110M | Unsupervised/Continuous Pre-trained |


### The Legacy Control: DistilBERT-v4

It serves as our historical benchmark. Trained during the early waves of dense text retrieval on the MS-MARCO dataset, it lacks modern contrastive training advancements and instruction tuning. Including it demonstrates exactly how much retrieval technology has progressed over the last few years.

### The Asymmetric Giants: BGE (Base vs. Large)

BAAI's BGE family introduces explicit task instructions (e.g., prepending query formatting strings). This allows the model to alter its vector mapping properties depending on whether it is processing a short search query or a long abstract.

The Scale Test: By evaluating both base and large, we explicitly test whether jumping from 110 million parameters to 326 million parameters yields enough of an accuracy boost to justify the dramatic surge in indexing and hardware costs.

### The Prefix Enforcers: E5 (Base vs. Large)

The E5 family utilizes a unique training format requiring strict structural prefixes (query:  and passage: ). It enforces an explicit geometric separation in the latent space between user intent and corpus data. Testing E5 directly against BGE provides a head-to-head match between the two most dominant open-source retrieval frameworks available.

### The Structural Alternative: Contriever

Developed by Meta, Contriever avoids traditional sentence-transformer pipelines. It utilizes a continuous pre-training approach optimized specifically for capturing dense, token-level matching structures without losing raw contextual data. This acts as an excellent mid-point between rigid keyword search (BM25) and highly abstract semantic models (BGE).


## 📊 Evaluation Results Summary

Our evaluation pipeline dynamically compiled 7,500 distinct document predictions across all 300 queries, comparing 25 unique system tracks.


| Search Approach | Total True Hits | Hit Rate | Precision@10 | Recall@10 | MRR@10 | NDCG@10 | Search Latency (ms/query) | Corpus Setup Overhead (sec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ANN Flat Cluster (BGE-Base-v1.5) | 294 | 0.98 | 0.0980 | 0.8767 | 0.6963 | 0.7386 | 29.32 | 761.61 |
| | | | | | | | | | 
| Brute Force (BGE-Large-v1.5) | 298 | 0.99 | 0.0993 | 0.8800 | 0.7124 | 0.7499 | 86.45 | 2734.36 |
| Brute Force (BGE-Base-v1.5) | 296 | 0.99 | 0.0987 | 0.8833 | 0.7034 | 0.7456 | 28.30 | 761.61 |
| HNSW Graph (BGE-Base-v1.5) | 296 | 0.99 | 0.0987 | 0.8833 | 0.7036 | 0.7458 | 29.05 | 761.61 |
| HNSW Graph (BGE-Large-v1.5) | 296 | 0.99 | 0.0987 | 0.8733 | 0.7057 | 0.7433 | 81.62 | 2734.36 |
| ANN Flat Cluster (BGE-Large-v1.5) | 293 | 0.98 | 0.0977 | 0.8667 | 0.7084 | 0.7437 | 82.38 | 2734.36 |
| HNSW Graph (E5-Large-v2) | 287 | 0.96 | 0.0957 | 0.8600 | 0.6881 | 0.7284 | 76.30 | 2670.21 |
| Brute Force (E5-Base-v2) | 286 | 0.95 | 0.0953 | 0.8600 | 0.6840 | 0.7245 | 25.79 | 701.16 |
| Brute Force (E5-Large-v2) | 286 | 0.95 | 0.0953 | 0.8567 | 0.6878 | 0.7274 | 75.49 | 2670.21 |
| HNSW Graph (E5-Base-v2) | 285 | 0.95 | 0.0950 | 0.8567 | 0.6810 | 0.7215 | 27.14 | 701.16 |
| ANN Flat Cluster (E5-Base-v2) | 280 | 0.93 | 0.0933 | 0.8400 | 0.6714 | 0.7103 | 27.81 | 701.16 |
| ANN Flat Cluster (E5-Large-v2) | 276 | 0.92 | 0.0920 | 0.8233 | 0.6599 | 0.6981 | 76.33 | 2670.21 |
| Hybrid (E5-Large-v2 + BM25) | 266 | 0.89 | 0.0887 | 0.8100 | 0.6310 | 0.6719 | 78.99 | 2670.21 |
| Brute Force (Contriever) | 265 | 0.88 | 0.0883 | 0.7967 | 0.6207 | 0.6601 | 26.12 | 817.07 |
| Hybrid (BGE-Base-v1.5 + BM25) | 265 | 0.88 | 0.0883 | 0.8067 | 0.6308 | 0.6715 | 30.90 | 761.61 |
| Hybrid (E5-Base-v2 + BM25) | 265 | 0.88 | 0.0883 | 0.8067 | 0.6337 | 0.6732 | 29.04 | 701.16 |
| HNSW Graph (Contriever) | 263 | 0.88 | 0.0877 | 0.7900 | 0.6196 | 0.6576 | 26.98 | 817.07 |
| Hybrid (BGE-Large-v1.5 + BM25) | 261 | 0.87 | 0.0870 | 0.7933 | 0.6381 | 0.6730 | 84.18 | 2734.36 |
| ANN Flat Cluster (Contriever) | 255 | 0.85 | 0.0850 | 0.7633 | 0.5915 | 0.6299 | 26.41 | 817.07 |
| Hybrid (Contriever + BM25) | 252 | 0.84 | 0.0840 | 0.7667 | 0.6145 | 0.6490 | 28.92 | 817.07 |
| Hybrid (DistilBERT-v4 + BM25) | 236 | 0.79 | 0.0787 | 0.7233 | 0.5657 | 0.6031 | 16.19 | 352.29 |
| BM25 Lexical Only | 229 | 0.76 | 0.0763 | 0.7033 | 0.5242 | 0.5678 | 12.76 | 0.00 |
| Brute Force (DistilBERT-v4) | 220 | 0.73 | 0.0733 | 0.6667 | 0.5009 | 0.5405 | 14.11 | 352.29 |
| HNSW Graph (DistilBERT-v4) | 218 | 0.73 | 0.0727 | 0.6600 | 0.4960 | 0.5352 | 14.20 | 352.29 |
| ANN Flat Cluster (DistilBERT-v4) | 208 | 0.69 | 0.0693 | 0.6267 | 0.4677 | 0.5059 | 14.32 | 352.29 |

---

## 💡 Key Engineering Takeaways

### 1. Model Selection & Efficiency (BGE-Base vs. BGE-Large)

While BGE-Large-v1.5 yields the highest raw accuracy (0.7499 NDCG@10), it introduces an unnecessary trade-off in this environment. BGE-Base-v1.5 achieves nearly identical accuracy (0.7458 NDCG@10 via HNSW) while delivering a 3x reduction in query latency (29.05ms) and saving significantly on localized corpus setup overhead. For production pipelines, base-sized models represent the optimal performance-to-compute threshold.


### 2. The Hybrid Fusion Paradox (RRF Dilution)

Combining keyword matching (BM25) and vector search via Reciprocal Rank Fusion (RRF) does not universally guarantee better results:

The Upgrade: For weaker, legacy semantic spaces like DistilBERT, hybrid fusion significantly elevated performance (NDCG rose from 0.5405 to 0.6031) by anchoring the search to exact medical keyword matches.

The Dilution: For highly optimized retrieval models (BGE, E5), standalone semantic capabilities are vastly superior to pure lexical matching (0.74 vs 0.56 NDCG). Forcing RRF to merge pristine semantic vectors with lower-ranked keyword results introduces noise, pulling down the top-tier rankings.

### 3. Approximate Nearest Neighbor (ANN) Fidelity

At a production scale, brute-force vector search ($O(N)$ complexity) becomes unviable. Benchmarking FAISS’s IndexHNSWFlat configuration demonstrates that approximate index graphs preserve near-perfect mathematical fidelity to exact brute-force calculations. For instance, BGE-Base on HNSW matched exact Brute Force precisely at 0.7458 NDCG@10, confirming that approximate index strategies can be deployed without degrading retrieval quality.

### 4. Generational Gap in Embeddings

The data highlights a sharp divide between generic text encoders and models explicitly fine-tuned on asymmetric text-retrieval tasks. Specialized search models (BGE, E5) vastly outperformed general-purpose legacy models—standalone BGE-Base achieved a 0.99 Hit Rate compared to DistilBERT at 0.73, proving that structural retrieval pre-training matters far more than raw vector dimensionality.



## 🛠️ System Architecture & Search Tracks

The project executes and logs four parallel retrieval pipelines over the corpus:

```text
                  ┌──► BM25 Lexical Only (Raw Term Frequency Inverse Document Frequency Baseline)
                  ├──► Brute Force (PyTorch Matrix Multiplication with L2 Normalization)
[User Query] ─────┼──► ANN Flat Cluster (FAISS Index IVF Flat Quantizer; nlist=64, nprobe=16)
                  ├──► HNSW Graph (FAISS Index HNSW Flat Hierarchical Network Graph)
                  └──► Hybrid Track ──► [HNSW + BM25Okapi] ──► Reciprocal Rank Fusion (RRF)
```

1. **BM25 Lexical Only:** Evaluates exact term frequencies matching using the `BM25Okapi` sparse tokens matrix on CPU.
2. **Brute Force (Exact Cosine):** Runs exact matrix tensor multiplications utilizing native `torch.mm` across Normalized Document Vectors on GPU/CPU. Serves as our control baseline.
3. **ANN Flat Cluster (`Index IVF Flat`):** Partitioning index that groups the normalized vector space into 64 distinct Voronoi cells. It accelerates lookups by restricting the search path to the nearest 16 clusters (`nprobe=16`).
4. **HNSW Graph (`Index HNSW Flat`):** Constructs a multi-layer hierarchical network graph (`M=32`, `efSearch=64`, `efConstruction=64`) for accelerated vector routing, yielding optimal approximate nearest neighbors.
5. **Hybrid (HNSW + BM25):** Executes a concurrent lexical-semantic pipeline. Exact keyword matching from `BM25Okapi` is mathematically merged with structural graphs from HNSW using **Reciprocal Rank Fusion (RRF)**:
   
$$RRF\_Score(d \in D) = \frac{1}{k_{rrf} + \text{rank}_{HNSW}(d)} + \frac{1}{k_{rrf} + \text{rank}_{BM25}(d)}$$


### 📈 Structural Mechanics of the Configured Indexes

#### 1. ANN Flat Cluster (`IndexIVFFlat`)
* **Partitioning Logic (`nlist=64`):** The vector space is partitioned into **64 distinct Voronoi cells** using k-means clustering. This mapping collapses an $O(N)$ exhaustive search down to an optimized sub-linear routing path.
* **Search Boundary (`nprobe=16`):** During retrieval, the engine isolates the query vector's location and restricts its search to only the **16 nearest clusters**. Modifying this single parameter via the configuration file allows engineers to dynamically navigate the precision-recall trade-off curve in production.

#### 2. HNSW Graph (`IndexHNSWFlat`)
* **Multi-Layer Routing Graph:** Constructs an advanced hierarchical structural network for non-exhaustive geometric traversal.
* **Graph Hyperparameters:** Configured out-of-the-box with link-allocation limits (`M=32`), index build verification checkpoints (`efConstruction=64`), and a fast runtime evaluation window (`efSearch=64`) to maximize vector routing fidelity.
* **Retrieval Depth (`top_k=10`):** The final evaluation pipeline truncates candidates exactly at the 10th neighbor to calculate localized ranking attributes like NDCG@10 and Recall@10.

---

## 📝 Metric Definitions for RAG Ingestion

The Author truncate all metrics strictly to the Top 10 results (`@10`) because it matches the token limits, API budgets, and practical context constraints of modern LLM prompt engineering.

* **Recall@10 (Quantity Metric):** Measures if the ground-truth document was successfully caught anywhere inside the Top 10 window. Critical for preventing **LLM Hallucinations**.
* **Precision@10 (Signal-to-Noise Metric):** Measures the proportion of helpful vs. junk documents entering the prompt. High precision keeps prompts clean and limits token billing.
* **MRR@10 & NDCG@10 (Sorting Quality Metrics):** Tracks positional accuracy. LLMs suffer from "lost-in-the-middle" bias and pay the closest attention to information at the absolute top of their prompt window. High MRR and NDCG verify that the strongest evidence is consistently delivered at **Rank 1 or Rank 2**.



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
```

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

4. Search Index Configurations & Parameter Tuning

The architecture decouples configuration from execution logic. Index hyperparameters, grouping scales, and retrieval bounds are dynamically controlled via a centralized configuration file (`config.py`), allowing immediate tuning without code modifications.

