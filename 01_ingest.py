import fitz
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from collections import Counter

PDF_DIR   = "./Dataset"
DB_PATH   = "./agri_db"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50

# ── Content-based crop detection (scans the chunk text itself) ──
CROP_KEYWORDS = {
    "rice": ["rice", "paddy", "basmati", "dsr"],
    "wheat": ["wheat", "triticum", "barley"],
    "maize": ["maize", "corn", "bajra"],
    "cotton": ["cotton", "gossypium", "kapas"],
    "chilli": ["chilli", "chili", "capsicum annuum"],
    "tomato": ["tomato", "lycopersicon"],
    "brinjal": ["brinjal", "eggplant", "brinjal"],
    "groundnut": ["groundnut", "peanut", "arachis"],
    "soyabean": ["soyabean", "soybean", "glycine max"],
    "sugarcane": ["sugarcane", "sugar cane", "saccharum"],
    "sorghum": ["sorghum", "jowar"],
    "millets": ["millet", "cumbu", "ragi", "finger millet", "pearl millet", "foxtail"],
    "redgram": ["redgram", "pigeonpea", "pigeon pea", "arhar", "tur"],
    "blackgram": ["blackgram", "black gram", "urad"],
    "greengram": ["greengram", "green gram", "moong", "mung"],
    "cowpea": ["cowpea", "cow pea"],
    "horsegram": ["horsegram", "horse gram"],
    "lentil": ["lentil", "masoor"],
    "sesame": ["sesame", "gingelly", "til"],
    "sunflower": ["sunflower"],
    "castor": ["castor"],
    "coconut": ["coconut"],
    "mustard": ["mustard", "rapeseed"],
    "safflower": ["safflower"],
    "mango": ["mango", "mangifera"],
    "banana": ["banana", "plantain", "musa"],
    "tomato": ["tomato"],
    "potato": ["potato", "solanum tuberosum"],
    "onion": ["onion", "allium"],
    "turmeric": ["turmeric", "curcuma"],
    "ginger": ["ginger", "zingiber"],
    "pepper": ["black pepper", "piper nigrum"],
    "cardamom": ["cardamom", "elettaria"],
    "coffee": ["coffee", "coffea"],
    "tea": ["tea", "camellia sinensis"],
    "cashew": ["cashew"],
    "rubber": ["rubber", "hevea"],
    "jute": ["jute", "corchorus"],
    "tobacco": ["tobacco", "nicotiana"],
    "mentha": ["mentha", "mint", "spearmint"],
    "lemongrass": ["lemongrass", "lemon grass"],
    "fodder": ["fodder", "napier", "lucerne", "berseem", "silage"],
    "ipm": ["integrated pest management", "ipm", "biocontrol", "biological control"],
}

# Filename-level fallback tag (used as doc_type, not crop)
FILENAME_TAG_MAP = {
    "horticulture": "horticulture-guide",
    "agriculture": "agriculture-guide",
    "angrau": "angrau-guide",
    "niphm": "niphm-ipm",
    "pp_kharif": "kharif-guide",
    "pp_rabi": "rabi-guide",
    "crop-kharif": "kharif-guide",
    "field-crop-kharif": "kharif-guide",
    "field-crops-rabi": "rabi-guide",
    "districtwise": "cotton-guide",
    "ipmmaize": "maize-ipm",
}

def get_doc_type(filename):
    fname = filename.lower().replace(" ", "-").replace("_", "-")
    for key, val in FILENAME_TAG_MAP.items():
        if key in fname:
            return val
    # fallback: use the crop name from filename
    fname2 = filename.lower()
    for crop in CROP_KEYWORDS:
        if crop in fname2:
            return f"{crop}-specific"
    return "general"

def detect_crops_in_chunk(text):
    """Return a list of all crops mentioned in this chunk's text."""
    text_lower = text.lower()
    found = []
    for crop, keywords in CROP_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(crop)
                break  # only add crop once even if multiple keywords match
    return found if found else ["general"]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        try:
            text = page.get_text()
            if text.strip():
                pages.append((page_num + 1, text))
        except Exception:
            pass
    return pages

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "]
)

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Wipe old DB and start fresh ──────────────────────────
print("Clearing old ChromaDB collection...")
client = chromadb.PersistentClient(path=DB_PATH)
try:
    client.delete_collection("agri_knowledge")
    print("  Old collection deleted.")
except Exception:
    print("  No old collection found, starting fresh.")

collection = client.get_or_create_collection(
    name="agri_knowledge",
    metadata={"hnsw:space": "cosine"}
)

all_docs, all_metas, all_ids, all_embeds = [], [], [], []
chunk_id = 0

if not os.path.exists(PDF_DIR):
    print(f"ERROR: Folder '{PDF_DIR}' not found!")
    exit()

pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
print(f"\nFound {len(pdf_files)} PDFs. Starting ingestion...\n")

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    doc_type = get_doc_type(pdf_file)
    before   = chunk_id

    print(f"Processing: {pdf_file}")
    print(f"  doc_type: {doc_type}")

    try:
        pages = extract_text_from_pdf(pdf_path)
        for page_num, page_text in pages:
            chunks = splitter.split_text(page_text)
            for chunk in chunks:
                if len(chunk.strip()) < 50:
                    continue

                # detect ALL crops mentioned in this specific chunk
                crops_in_chunk = detect_crops_in_chunk(chunk)
                primary_crop   = crops_in_chunk[0]
                all_crops_str  = ",".join(crops_in_chunk)  # e.g. "rice,ipm,general"

                embedding = embedder.encode(chunk).tolist()
                all_docs.append(chunk)
                all_metas.append({
                    "crop":      primary_crop,   # primary — used for filtering
                    "all_crops": all_crops_str,  # all detected — for reference
                    "doc_type":  doc_type,
                    "filename":  pdf_file,
                    "page":      page_num,
                })
                all_ids.append(f"chunk_{chunk_id}")
                all_embeds.append(embedding)
                chunk_id += 1

        print(f"  → {chunk_id - before} chunks added (total: {chunk_id})")
    except Exception as e:
        print(f"  SKIPPED — error: {e}")

print(f"\nSaving {chunk_id} chunks to ChromaDB...")
BATCH = 500
for i in range(0, len(all_docs), BATCH):
    collection.add(
        documents  = all_docs[i:i+BATCH],
        metadatas  = all_metas[i:i+BATCH],
        ids        = all_ids[i:i+BATCH],
        embeddings = all_embeds[i:i+BATCH],
    )
    print(f"  Saved {i} → {min(i+BATCH, len(all_docs))}")

# ── Final summary ────────────────────────────────────────
print(f"\n{'='*50}")
print(f"DONE! {chunk_id} total chunks stored.")
print(f"{'='*50}")
print("\nChunks per PRIMARY crop tag:")
tag_counts = Counter(m["crop"] for m in all_metas)
for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
    bar = "█" * (count // 100)
    print(f"  {tag:25s} {count:>5}  {bar}")

print("\nChunks per doc_type:")
doc_counts = Counter(m["doc_type"] for m in all_metas)
for tag, count in sorted(doc_counts.items(), key=lambda x: -x[1]):
    print(f"  {tag:30s} {count:>5}")