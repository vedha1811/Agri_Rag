import math
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from config import DB_PATH, TOP_K, RERANK_TOP

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Connecting to ChromaDB...")
client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("agri_knowledge")
print(f"Ready! {collection.count()} chunks loaded.\n")


def normalize_score(score):
    return round(100 / (1 + math.exp(-score)), 1)


def clean_chunk(text):
    """Clean up a chunk — remove lone bullet points and fix spacing."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        # Skip lines that are just a bullet point with nothing else
        if line in ['•', '-', '*', '·', '–']:
            continue
        if line:
            cleaned.append(line)
    return ' '.join(cleaned).strip()


def is_good_chunk(text):
    """Return True if chunk seems complete enough to show."""
    text = text.strip()
    if len(text) < 60:
        return False
    # Skip chunks that start with lowercase (mid-sentence break)
    first_word = text.split()[0] if text.split() else ''
    if first_word and first_word[0].islower() and len(text) < 150:
        return False
    return True


def retrieve(query, crop_filter=None):
    """Embed query → semantic search → rerank → return top results."""
    query_embed  = embedder.encode(query).tolist()
    where_clause = {"crop": crop_filter} if crop_filter and crop_filter != "any" else None

    # Fetch more candidates so we have enough after filtering
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=TOP_K * 2,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )

    candidates = results["documents"][0]
    metadatas  = results["metadatas"][0]

    if not candidates:
        return []

    # Rerank all candidates
    pairs  = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates, metadatas),
        key=lambda x: x[0],
        reverse=True
    )

    # Clean and filter chunks, return top RERANK_TOP good ones
    final = []
    for score, doc, meta in ranked:
        cleaned = clean_chunk(doc)
        if is_good_chunk(cleaned):
            final.append((score, cleaned, meta))
        if len(final) >= RERANK_TOP:
            break

    # Fallback: if filtering removed too many, return top ranked anyway
    if len(final) < 2:
        final = [(s, clean_chunk(d), m) for s, d, m in ranked[:RERANK_TOP]]

    return final