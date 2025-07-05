import streamlit as st
import re
from rank_bm25 import BM25Okapi
from bs4 import BeautifulSoup

# ---------- Load Data ----------
def parse_corpus_trec(file_path):
    corpus = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    for doc in soup.find_all('doc'):
        doc_id = doc.docno.text.strip()
        body = doc.body.text.strip()
        corpus[doc_id] = body
    return corpus

def tokenize(text):
    return re.findall(r'\w+', text.lower())

@st.cache_data

def load_bm25(corpus):
    tokenized_corpus = [tokenize(doc) for doc in corpus.values()]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, list(corpus.keys()), list(corpus.values())

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Code-Mixed IR", layout="wide")
st.title("🔍 Bengali-English Code-Mixed Search")
st.write("Search through code-mixed social media content using BM25 retrieval.")

query = st.text_input("Enter your query (code-mixed):", "hyderabad to howrah kono train ache?")

if query:
    with st.spinner("Retrieving relevant documents..."):
        corpus = parse_corpus_trec('data/corpus_trec.txt')
        bm25, doc_ids, doc_texts = load_bm25(corpus)

        scores = bm25.get_scores(tokenize(query))
        top_k = sorted(zip(doc_ids, doc_texts, scores), key=lambda x: x[2], reverse=True)[:10]

        st.subheader(f"Top {len(top_k)} Relevant Results")
        for i, (doc_id, body, score) in enumerate(top_k, 1):
            st.markdown(f"**{i}. Doc ID: {doc_id} | Score: {score:.4f}**")
            st.write(body)
            st.markdown("---")
