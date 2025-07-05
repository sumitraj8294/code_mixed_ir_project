import os
import re
from rank_bm25 import BM25Okapi
from bs4 import BeautifulSoup

# ----------- Utility Functions ------------
def parse_corpus_trec(file_path):
    corpus = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    for doc in soup.find_all('doc'):
        doc_id = doc.docno.text.strip()
        body = doc.body.text.strip()
        corpus[doc_id] = body
    return corpus

def parse_queries_trec(file_path):
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    for top in soup.find_all('top'):
        qid = top.num.text.strip()
        title = top.title.text.strip().replace('\n', ' ')
        queries[qid] = title
    return queries

def tokenize(text):
    return re.findall(r'\w+', text.lower())

# ----------- BM25 Retrieval ------------
def retrieve_top_k(corpus, queries, k=10):
    tokenized_corpus = [tokenize(corpus[doc_id]) for doc_id in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_ids = list(corpus.keys())

    results = []
    for qid, query in queries.items():
        scores = bm25.get_scores(tokenize(query))
        ranked_docs = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:k]
        for rank, (doc_id, score) in enumerate(ranked_docs):
            results.append(f"{qid} Q0 {doc_id} {rank+1} {score:.4f} BM25")
    return results

# ----------- Main Execution ------------
if __name__ == '__main__':
    corpus_path = 'data/corpus_trec.txt'
    queries_path = 'data/queries_trec.txt'
    output_path = 'runs/bm25_run.txt'

    os.makedirs('runs', exist_ok=True)

    corpus = parse_corpus_trec(corpus_path)
    queries = parse_queries_trec(queries_path)

    bm25_results = retrieve_top_k(corpus, queries, k=100)

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in bm25_results:
            f.write(line + '\n')

    print(f"BM25 retrieval completed. Results saved to {output_path}")
