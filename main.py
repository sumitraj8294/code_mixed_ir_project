from retrievers.bm25_retriever import build_bm25, retrieve_top_k, save_results
from utils.loader import load_trec_corpus, load_trec_queries
from evaluation.evaluate import evaluate_run
import os

def run_bm25_pipeline():
    corpus = load_trec_corpus("data/corpus_trec.txt")
    queries = load_trec_queries("data/queries_trec.txt")
    bm25, doc_ids, _ = build_bm25(corpus)

    results = retrieve_top_k(bm25, doc_ids, queries, k=10)
    os.makedirs("runs", exist_ok=True)
    result_file = "runs/bm25_run.txt"
    save_results(results, result_file)
    print("BM25 run saved!")

    metrics = evaluate_run(result_file, "data/qrels_train.txt", k=10)
    print("Evaluation Metrics:")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")

if __name__ == "__main__":
    run_bm25_pipeline()
