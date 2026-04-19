from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "chroma_db"
OUTPUTS_DIR = BASE_DIR / "outputs"
QUESTIONS_FILE = BASE_DIR / "configs" / "questions.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
FIXED_CHUNK_SIZE = 250
FIXED_CHUNK_OVERLAP = 20
RECURSIVE_CHUNK_SIZE = 400
RECURSIVE_CHUNK_OVERLAP = 80

FIXED_COLLECTION = "personal_knowledge_fixed"
RECURSIVE_COLLECTION = "personal_knowledge_recursive"

# ── Multi-turn memory settings ──────────────────────────────────
MEMORY_MAX_TURNS_BEFORE_SUMMARY = 10   # compress history after this many Q&A pairs
MEMORY_SUMMARY_KEEP_LAST_N = 4         # keep this many recent turns verbatim
MEMORY_SIMILARITY_THRESHOLD = 0.82     # cosine sim to flag a duplicate question