import fitz
import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from collections import Counter

DB_PATH   = "./agri_db"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50

# Only the scanned PDFs
SCANNED_PDFS = [
    ("Dataset/wheat.pdf",                                    "wheat"),
    ("Dataset/ANGRAU_Technologies developed final copy.pdf", "angrau-guide"),
]

CROP_KEYWORDS = {
    "rice": ["rice", "paddy", "basmati", "dsr"],
    "wheat": ["wheat", "triticum", "barley", "roti", "chapati"],
    "maize": ["maize", "corn", "bajra"],
    "cotton": ["cotton", "gossypium", "kapas"],
    "chilli": ["chilli", "chili", "capsicum annuum"],
    "tomato": ["tomato", "lycopersicon"],
    "brinjal": ["brinjal", "eggplant"],
    "groundnut": ["groundnut", "peanut", "arachis"],
    "soyabean": ["soyabean", "soybean", "glycine max"],
    "sugarcane": ["sugarcane", "sugar cane", "saccharum"],
    "sorghum": ["sorghum", "jowar"],
    "millets": ["millet", "cumbu", "ragi", "finger millet", "pearl millet"],
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
    "groundnut": ["groundnut", "peanut"],
    "mango": ["mango", "mangifera"],
    "banana": ["banana", "plantain"],
    "potato": ["potato", "solanum tuberosum"],
    "onion": ["onion", "allium"],
    "turmeric": ["turmeric", "curcuma"],
    "ginger": ["ginger", "zingiber"],
    "fodder": ["fodder", "napier", "lucerne", "berseem", "silage"],
    "ipm": ["integrated pest management", "ipm", "biocontrol"],
}

def detect_crops_in_chunk(text):
    text_lower = text.lower()
    found = []
    for crop, keywords in CROP_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(crop)
                break
    return found if found else ["general"]

def ocr_pdf(pdf_path):
    """Convert each PDF page to image, then run Tesseract OCR on it."""
    print(f"  Converting pages to images...")
    pages_images = convert_from_path(pdf_path, dpi=200)
    print(f"  Running OCR on {len(pages_images)} pages...")

    pages_text = []
    for i, img in enumerate(pages_images):
        text = pytesseract.image_to_string(img, lang="eng")
        if text.strip():
            pages_text.append((i + 1, text))
        if (i + 1) % 10 == 0:
            print(f"    OCR progress: {i+1}/{len(pages_images)} pages done")
    return pages_text

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "]
)

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Connect to EXISTING ChromaDB (don't wipe it!) ────────
client     = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="agri_knowledge",
    metadata={"hnsw:space": "cosine"}
)

print(f"Existing chunks in DB: {collection.count()}")

all_docs, all_metas, all_ids, all_embeds = [], [], [], []

# Get the highest existing chunk_id so we don't overwrite
existing = collection.get(include=[])
existing_ids = existing["ids"]
if existing_ids:
    max_id = max(int(i.split("_")[1]) for i in existing_ids)
else:
    max_id = -1
chunk_id = max_id + 1
print(f"Starting new chunks from ID: chunk_{chunk_id}\n")

for pdf_path, doc_type in SCANNED_PDFS:
    print(f"\nProcessing (OCR): {pdf_path}")
    print(f"  doc_type: {doc_type}")
    before = chunk_id

    try:
        pages = ocr_pdf(pdf_path)
        for page_num, page_text in pages:
            chunks = splitter.split_text(page_text)
            for chunk in chunks:
                if len(chunk.strip()) < 50:
                    continue
                crops_in_chunk = detect_crops_in_chunk(chunk)
                primary_crop   = crops_in_chunk[0]
                all_crops_str  = ",".join(crops_in_chunk)

                embedding = embedder.encode(chunk).tolist()
                all_docs.append(chunk)
                all_metas.append({
                    "crop":      primary_crop,
                    "all_crops": all_crops_str,
                    "doc_type":  doc_type,
                    "filename":  os.path.basename(pdf_path),
                    "page":      page_num,
                })
                all_ids.append(f"chunk_{chunk_id}")
                all_embeds.append(embedding)
                chunk_id += 1

        print(f"  → {chunk_id - before} OCR chunks extracted")
    except Exception as e:
        print(f"  FAILED — {e}")

if all_docs:
    print(f"\nSaving {len(all_docs)} new OCR chunks to ChromaDB...")
    BATCH = 500
    for i in range(0, len(all_docs), BATCH):
        collection.add(
            documents  = all_docs[i:i+BATCH],
            metadatas  = all_metas[i:i+BATCH],
            ids        = all_ids[i:i+BATCH],
            embeddings = all_embeds[i:i+BATCH],
        )
        print(f"  Saved {i} → {min(i+BATCH, len(all_docs))}")

    print(f"\nDone! Total chunks in DB now: {collection.count()}")
else:
    print("No new chunks extracted — check if Tesseract is installed correctly.")