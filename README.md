# Semantic Search Tool

A semantic search engine that retrieves documents by **meaning**, not by keyword matching.

Given a natural-language query, the tool returns the most semantically relevant documents from a corpus, even when the query and the documents share no exact words.

## How It Works

```text
Documents ──► Preprocess ──► Sentence Transformer ──► Document Embeddings
                                                              │
                                                              ▼
                                                        Vector Index
                                                              ▲
                                                              │
User Query ──► Sentence Transformer ──► Query Embedding ──► Cosine Similarity
                                                              │
                                                              ▼
                                                        Rank + Top-K
                                                              │
                                                              ▼
                                                           Results
```

1. Each document is cleaned and converted into a dense vector (embedding) using a sentence transformer.
2. The same model embeds the user's query into the same vector space.
3. Cosine similarity scores every document against the query.
4. The top-K highest-scoring documents are returned.

## Example

**Query:** `How do computers learn from information?`

**Results:**

```text
[0.6238] Machine learning allows computers to learn patterns from data.
[0.4861] Natural language processing helps computers understand human language.
[0.3993] Neural networks are inspired by the human brain.
```

The query shares no exact keywords with the top result. The match is based on meaning.

## Tech Stack

- **Python 3.11**
- **sentence-transformers**: `all-MiniLM-L6-v2` model (384-dim embeddings)
- **NumPy**: vector operations
- **scikit-learn**: cosine similarity
- **Optional:** FAISS for large-scale vector search

## Project Structure

```text
semantic_search/
├── documents.txt        # one sentence per line, the searchable corpus
├── semantic_search.py   # main script
├── requirements.txt     # dependencies
└── .gitignore
```

## Setup

```bash
git clone https://github.com/IIGGRRIISS/semantic-search.git
cd semantic-search

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

```bash
python semantic_search.py
```

You'll see an interactive prompt:

```text
============================================================
Semantic Search Ready
Indexed 2000 documents.
Type a query and press Enter. Type 'quit' to exit.
============================================================

Query: who invented the telephone
------------------------------------------------------------
1. [0.7214] Alexander Graham Bell is commonly credited with inventing the telephone.
2. [0.5123] The telephone was invented in 1876.
3. [0.4401] Email was invented in the early 1970s.
```

## Features

- **Meaning-based retrieval:** finds relevant results even without keyword overlap
- **Same model for documents and queries:** guarantees both live in the same vector space
- **Embedding cache:** the first run indexes the corpus; later runs load instantly from `embeddings.npz`
- **Interactive CLI:** type queries in a loop, `quit` to exit
- **Top-K ranking:** returns the K highest-similarity documents

## Configuration

Change the corpus by editing `documents.txt` (one document per line).

Change the model in `semantic_search.py`:

```python
engine = SemanticSearch(model_name="all-MiniLM-L6-v2")
```

Other options: `all-mpnet-base-v2` (higher quality, slower) and `paraphrase-MiniLM-L3-v2` (faster, smaller).

## Why Semantic Search?

Traditional keyword search fails when the query and document use different words for the same concept:

| Query                            | Document                                                      | Keyword match | Semantic match |
|----------------------------------|---------------------------------------------------------------|---------------|----------------|
| "How do computers learn?"        | "Machine learning allows computers to learn patterns from data." | ❌            | ✅             |
| "What language is used for AI?"  | "Python is commonly used for machine learning."               | ❌            | ✅             |
| "Popular sport worldwide"        | "Football is the world's most popular sport."                 | ⚠️ partial    | ✅             |

Semantic search solves this by comparing meaning, not strings.

## Limitations

- Retrieval quality depends on the embedding model.
- Very long documents should be split into sentences or chunks before indexing.
- No approximate nearest-neighbor index: brute-force cosine similarity is used, which is fast up to ~100k documents. Add FAISS beyond that.

## License

MIT

## Author

**Syed Ibrahim Ali**
GitHub: [IIGGRRIISS](https://github.com/IIGGRRIISS)
