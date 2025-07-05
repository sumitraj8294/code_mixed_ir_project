import os
from collections import defaultdict

# ----------- Load Qrels ------------
def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qrels[qid][docid] = int(rel)
    return qrels

# ----------- Load Run File ------------
def load_run(path):
    run = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, method = line.strip().split()
            run[qid].append(docid)
    return run

# ----------- Evaluation Metrics ------------
def precision_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / k

def recall_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / len(relevant) if relevant else 0

def average_precision(relevant, retrieved):
    hits = 0
    sum_precisions = 0
    for i, docid in enumerate(retrieved):
        if docid in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant) if relevant else 0

def ndcg_at_k(relevant, retrieved, k):
    def dcg(rels):
        return sum([rel / (log2(i + 2)) for i, rel in enumerate(rels)])

    def log2(x):
        from math import log2 as _log2
        return _log2(x)

    retrieved_k = retrieved[:k]
    rel_scores = [1 if docid in relevant else 0 for docid in retrieved_k]
    ideal = sorted(rel_scores, reverse=True)
    return dcg(rel_scores) / dcg(ideal) if dcg(ideal) > 0 else 0

# ----------- Main Evaluation ------------
def evaluate(qrels_path, run_path, k=10):
    qrels = load_qrels(qrels_path)
    run = load_run(run_path)

    total_p, total_r, total_ap, total_ndcg = 0, 0, 0, 0
    count = 0

    for qid, retrieved_docs in run.items():
        relevant_docs = qrels.get(qid, {})
        relevant_set = set(doc for doc, rel in relevant_docs.items() if rel > 0)

        print(f"\nQuery: {qid}")
        print(f"Retrieved: {retrieved_docs[:10]}")
        print(f"Relevant: {list(relevant_set)}")

        total_p += precision_at_k(relevant_set, retrieved_docs, k)
        total_r += recall_at_k(relevant_set, retrieved_docs, k)
        total_ap += average_precision(relevant_set, retrieved_docs)
        total_ndcg += ndcg_at_k(relevant_set, retrieved_docs, k)
        count += 1

    print("\n\n========= Evaluation Results =========")
    print(f"MAP: {total_ap / count:.4f}")
    print(f"Precision@{k}: {total_p / count:.4f}")
    print(f"Recall@{k}: {total_r / count:.4f}")
    print(f"nDCG@{k}: {total_ndcg / count:.4f}")

if __name__ == '__main__':
    qrels_path = 'data/qrels_train.txt'
    run_path = 'runs/bm25_run.txt'
    evaluate(qrels_path, run_path, k=10)