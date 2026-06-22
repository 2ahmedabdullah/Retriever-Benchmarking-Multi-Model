# app.py
import logging
import sys
import gc
import torch
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from retrievers.dense import (
    compute_raw_embeddings,
    run_brute_force_retrieval,
    run_ann_retrieval,
    run_hnsw_lightning_retrieval,
    run_bm25_retrieval,
    run_hybrid_rrf_retrieval
)
from evaluation.metrics import get_predictions_dataframe

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)

DATA_PATH = "./datasets/scifact"

NLIST_PARAM = 64
NPROBE_PARAM = 16

TARGET_MODELS = {
    "DistilBERT-v4": "msmarco-distilbert-base-v4",
    "BGE-Base-v1.5": "BAAI/bge-base-en-v1.5",
    "BGE-Large-v1.5": "BAAI/bge-large-en-v1.5",
    "E5-Base-v2": "intfloat/e5-base-v2",
    "E5-Large-v2": "intfloat/e5-large-v2",
    "Contriever": "facebook/contriever-msmarco"
}

if __name__ == "__main__":
    print("Executing Master Pipeline Step 1: Ingesting Corpus into memory...", flush=True)
    corpus, queries, qrels = GenericDataLoader(data_folder=DATA_PATH).load(split="test")
    
    # Run BM25 baseline once
    bm25_results, bm25_lat = run_bm25_retrieval(corpus, queries)
    
    strategies = {"BM25 Lexical Only": bm25_results}
    latency_scorecard = [{
        "Strategy": "BM25 Lexical Only", 
        "Search Latency (ms/query)": round(bm25_lat, 4),
        "Corpus Setup Overhead (sec)": 0.0
    }]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for architecture_label, huggingface_id in TARGET_MODELS.items():
        try:
            print(f"\n=================== RUNNING: {architecture_label} ===================", flush=True)
            model = models.SentenceBERT(huggingface_id, device=device)
            
            # CRITICAL LAYER: Run raw encoding exactly ONCE per model, capture overhead time
            doc_embeddings, doc_ids, corpus_overhead_secs = compute_raw_embeddings(model, corpus, huggingface_id)
            
            # Execute search strategies using the pre-computed RAM tensors
            bf_res, bf_lat = run_brute_force_retrieval(model, doc_embeddings, doc_ids, queries, huggingface_id, device)
            
            flat_res, flat_lat = run_ann_retrieval(
                model=model, 
                doc_embeddings=doc_embeddings, 
                doc_ids=doc_ids, 
                queries=queries, 
                model_name=huggingface_id, 
                nlist=NLIST_PARAM, 
                nprobe=NPROBE_PARAM
            )
            
            hnsw_res, hnsw_lat = run_hnsw_lightning_retrieval(model, doc_embeddings, doc_ids, queries, huggingface_id)
            hybrid_res, hyb_lat = run_hybrid_rrf_retrieval(hnsw_res, bm25_results)
            
            # Map predictions to output structures
            strategies[f"Brute Force ({architecture_label})"] = bf_res
            strategies[f"ANN Flat Cluster ({architecture_label})"] = flat_res
            strategies[f"HNSW Graph ({architecture_label})"] = hnsw_res
            strategies[f"Hybrid ({architecture_label} + BM25)"] = hybrid_res
            
            # Record pure search latency alongside the setup overhead
            latency_scorecard.extend([
                {
                    "Strategy": f"Brute Force ({architecture_label})", 
                    "Search Latency (ms/query)": round(bf_lat, 4),
                    "Corpus Setup Overhead (sec)": round(corpus_overhead_secs, 2)
                },
                {
                    "Strategy": f"ANN Flat Cluster ({architecture_label})", 
                    "Search Latency (ms/query)": round(flat_lat, 4),
                    "Corpus Setup Overhead (sec)": round(corpus_overhead_secs, 2)
                },
                {
                    "Strategy": f"HNSW Graph ({architecture_label})", 
                    "Search Latency (ms/query)": round(hnsw_lat, 4),
                    "Corpus Setup Overhead (sec)": round(corpus_overhead_secs, 2)
                },
                {
                    "Strategy": f"Hybrid ({architecture_label} + BM25)", 
                    "Search Latency (ms/query)": round(hyb_lat + hnsw_lat, 4),
                    "Corpus Setup Overhead (sec)": round(corpus_overhead_secs, 2)
                }
            ])
            
        except Exception as e:
            print(f"\nCRITICAL FAULT encountered while running evaluation for {architecture_label}: {str(e)}")
        finally:
            if 'model' in locals():
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nStep 3: Exporting predictions into a unified data structure...", flush=True)
    master_df = get_predictions_dataframe(
        qrels=qrels, 
        strategies_dict=strategies, 
        output_path="./results/raw_search_predictions.xlsx",
        top_k=10
    )
    
    # Save latency performance sheet
    lat_df = pd.DataFrame(latency_scorecard)
    print("\n======================= SYSTEM LATENCY REPORT =======================")
    print(lat_df.to_string(index=False))
    print("=====================================================================")
    lat_df.to_excel("./results/system_latency_scorecard.xlsx", index=False)