# =========================================================
# Configuration — Agricultural RAG System
# =========================================================

# ===== ChromaDB =====
DB_PATH = "./agri_db"

# ===== Retrieval =====
TOP_K = 10
RERANK_TOP = 5

# =========================================================
# Available LLM Models
# Keys are internal identifiers
# Values are names shown in the UI
# =========================================================
MODELS = {

    "auto":
        "⭐ Best Available Model (Auto)",

    "groq-llama":
        "LLaMA 3.1 8B (Groq)",

    "cohere-command":
        "Command-R (Cohere)",

    "sarvam-ai":
         "Sarvam-105B (Sarvam AI)",

    "cerebras-llama":
        "GPT-OSS 120B (Cerebras)",

}

DEFAULT_MODEL = "auto"


CROP_CHOICES = sorted([

    "any",

    "rice",
    "wheat",
    "maize",
    "cotton",
    "chilli",
    "tomato",
    "brinjal",
    "groundnut",
    "soyabean",
    "sugarcane",
    "sorghum",

    "millets",

    "redgram",
    "blackgram",
    "greengram",
    "cowpea",
    "horsegram",
    "lentil",

    "sesame",
    "sunflower",
    "castor",
    "coconut",
    "mustard",

    "mango",
    "banana",

    "potato",
    "onion",

    "turmeric",
    "ginger",
    "pepper",
    "cardamom",

    "coffee",
    "tea",
    "cashew",
    "rubber",
    "jute",
    "tobacco",

    "mentha",
    "lemongrass",

    "fodder",

    "ipm"
])