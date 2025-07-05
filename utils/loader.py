def load_trec_corpus(path):
    corpus = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                doc_id, text = parts
                corpus[doc_id] = text
    return corpus

def load_trec_queries(path):
    queries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                query_id, query = parts
                queries[query_id] = query
    return queries

def load_qrels(path):
    qrels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            qrels.setdefault(query_id, {})[doc_id] = int(relevance)
    return qrels
