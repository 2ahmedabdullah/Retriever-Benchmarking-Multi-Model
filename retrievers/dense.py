# retrievers/dense.py
import torch
import numpy as np
import time
import faiss
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def _prepare_query_text(q_text, model_name):
    """Applies asymmetrical instructions to queries for specific model families."""
    model_lower = model_name.lower()
    if "bge-" in model_lower:
        return f"Represent this sentence for searching relevant passages: {q_text}"
    elif "e5-" in model_lower:
        return f"query: {q_text}"
    return q_text

def compute_raw_embeddings(model, corpus, model_name, batch_size=256):
    """
    IGNORES DISK CACHE EXACTLY ONCE.
    Encodes document texts into RAM from scratch at the start of the model's turn
    to capture the baseline model encoding overhead without polluting search track times.
    """
    print(f"--> [LATENCY TEST] Encoding corpus into RAM from scratch using {model_name}...", flush=True)
    doc_ids = list(corpus.keys())
    
    is_e5 = "e5-" in model_name.lower()
    doc_texts = []
    for cid in doc_ids:
        title = corpus[cid].get("title", "")
        text = corpus[cid]["text"]
        combined = f"{title} {text}".strip()
        if is_e5:
            combined = f"passage: {combined}"
        doc_texts.append({"text": combined})
        
    start_time = time.perf_counter()
    doc_embeddings = model.encode_corpus(doc_texts, batch_size=batch_size)
    elapsed = time.perf_counter() - start_time
    
    print(f"--> Corpus Encoding Completed in {elapsed:.3f} seconds.", flush=True)
    return doc_embeddings.astype('float32'), doc_ids, elapsed

def run_brute_force_retrieval(model, doc_embeddings, doc_ids, queries, model_name, device):
    print(f"Executing [Brute Force Matrix Run] for {model_name}...", flush=True)
    
    start_time = time.perf_counter()
    doc_embeddings_tensor = torch.from_numpy(doc_embeddings).to(device)
    norm_docs = torch.nn.functional.normalize(doc_embeddings_tensor, p=2, dim=1)
    
    results = {}
    for q_id, q_text in tqdm(queries.items(), desc="Brute Force Loop", unit="query", leave=False):
        processed_query = _prepare_query_text(q_text, model_name)
        q_emb = torch.from_numpy(model.encode_queries([processed_query], batch_size=1, show_progress_bar=False)).to(device)
        norm_q = torch.nn.functional.normalize(q_emb, p=2, dim=1)
        
        cos_scores = torch.mm(norm_q, norm_docs.transpose(0, 1)).squeeze(0).tolist()
        results[q_id] = {doc_ids[idx]: score for idx, score in enumerate(cos_scores)}
        
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_query = elapsed_ms / len(queries)
    return results, latency_per_query

def run_ann_retrieval(model, doc_embeddings, doc_ids, queries, model_name, nlist, nprobe):
    print(f"Executing [FAISS IndexIVFFlat Run] with nlist={nlist}, nprobe={nprobe} for {model_name}...", flush=True)
    
    start_time = time.perf_counter()
    dimension = doc_embeddings.shape[1]
    ann_embeddings = doc_embeddings.copy()
    faiss.normalize_L2(ann_embeddings)
    
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(ann_embeddings)
    index.add(ann_embeddings)
    index.nprobe = nprobe
    
    results = {}
    for q_id, q_text in tqdm(queries.items(), desc="ANN Flat Loop", unit="query", leave=False):
        processed_query = _prepare_query_text(q_text, model_name)
        q_emb = model.encode_queries([processed_query], batch_size=1, show_progress_bar=False).astype('float32')
        faiss.normalize_L2(q_emb)
        
        scores, indices = index.search(q_emb, len(doc_ids))
        results[q_id] = {doc_ids[idx]: float(score) for score, idx in zip(scores[0], indices[0]) if idx != -1}
        
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_query = elapsed_ms / len(queries)
    return results, latency_per_query

def run_hnsw_lightning_retrieval(model, doc_embeddings, doc_ids, queries, model_name, M=32):
    print(f"Executing [FAISS IndexHNSWFlat Graph Run] for {model_name}...", flush=True)
    
    start_time = time.perf_counter()
    dimension = doc_embeddings.shape[1]
    hnsw_embeddings = doc_embeddings.copy()
    faiss.normalize_L2(hnsw_embeddings)
    
    index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 64
    index.hnsw.efConstruction = 64
    index.add(hnsw_embeddings)
    
    results = {}
    for q_id, q_text in tqdm(queries.items(), desc="HNSW Loop", unit="query", leave=False):
        processed_query = _prepare_query_text(q_text, model_name)
        q_emb = model.encode_queries([processed_query], batch_size=1, show_progress_bar=False).astype('float32')
        faiss.normalize_L2(q_emb)
        
        scores, indices = index.search(q_emb, len(doc_ids))
        results[q_id] = {doc_ids[idx]: float(score) for score, idx in zip(scores[0], indices[0]) if idx != -1}
        
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_query = elapsed_ms / len(queries)
    return results, latency_per_query

def run_bm25_retrieval(corpus, queries):
    print(f"\n[STRATEGY: BM25 LEXICAL ONLY] Measuring keyword matching latency...", flush=True)
    
    start_time = time.perf_counter()
    doc_ids = list(corpus.keys())
    tokenized_corpus = [(corpus[cid].get("title", "") + " " + corpus[cid]["text"]).lower().split() for cid in doc_ids]
    bm25 = BM25Okapi(tokenized_corpus)
    
    results = {}
    for q_id, q_text in tqdm(queries.items(), desc="BM25 Search Loop", unit="query"):
        tokenized_query = q_text.lower().split()
        doc_scores = bm25.get_scores(tokenized_query)
        results[q_id] = {doc_ids[idx]: float(doc_scores[idx]) for idx in range(len(doc_ids))}
        
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_query = elapsed_ms / len(queries)
    return results, latency_per_query

def run_hybrid_rrf_retrieval(dense_results, bm25_results, k_rrf=60):
    start_time = time.perf_counter()
    hybrid_results = {}
    for q_id in dense_results.keys():
        dense_ranked = sorted(dense_results[q_id].items(), key=lambda x: x[1], reverse=True)
        bm25_ranked = sorted(bm25_results[q_id].items(), key=lambda x: x[1], reverse=True)
        
        rrf_scores = {}
        for rank, (doc_id, _) in enumerate(dense_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + (rank + 1)))
        for rank, (doc_id, _) in enumerate(bm25_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + (rank + 1)))
            
        hybrid_results[q_id] = rrf_scores
        
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_query = elapsed_ms / len(dense_results)
    return hybrid_results, latency_per_query