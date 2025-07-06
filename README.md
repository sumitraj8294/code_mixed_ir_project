# Code-Mixed IR System (Bengali-English in Roman Script)

This project implements an Information Retrieval (IR) system for code-mixed Bengali-English queries and documents, written in Roman script, as commonly used in Indian social media communities. The system retrieves relevant answers from a corpus using BM25 ranking.

---
**
**Dont forget to add your here in the data directory**

# ✅ Installation

1. Create and activate a virtual environment (optional but recommended):

python -m venv venv

venv\Scripts\activate  # On Windows

# source venv/bin/activate  # On Linux/Mac


2. Install required packages:

pip install -r requirements.txt

If you get an error for pytrec_eval, install Microsoft C++ Build Tools from:

https://visualstudio.microsoft.com/visual-cpp-build-tools/ ```


🚀 Usage

1️⃣ Run BM25 Retrieval

python bm25_retriever.py

Loads corpus and queries

Ranks documents using BM25


Outputs top results to runs/bm25_run.txt


2️⃣ Evaluate Retrieval Performance


python evaluation/evaluate.py

This uses pytrec_eval to compute metrics like MAP, nDCG, Precision@10, Recall@10


3️⃣ Launch Web Interface (Optional)


streamlit run bm25_web_app.py

This opens a web page to enter your own code-mixed queries and see ranked results.


🔧 Improvements To Explore

Add semantic re-ranking using SBERT or pyLLMSearch


Normalize Romanized spellings (e.g., ami, aami, amee)


Support transliteration back to native script (optional)



🙌 Acknowledgement

This project is based on real-world information needs of code-mixed communities during events like COVID-19. Special thanks to researchers and data annotators involved in query and qrels creation.


