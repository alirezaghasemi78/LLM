
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
LLM_PROVIDER = "gemini"
LLM_MODEL_NAME = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_TEMPERATURE = 0.2
LLM_MAX_OUTPUT_TOKENS = 1024

# Dataset settings
DATASET_NAME = "codersan/Persian-Wikipedia-Corpus"
DATASET_SPLIT = "train"

# Number of documents to load
MAX_DOCUMENTS = 1000


# Project paths
DATA_DIR = Path(
    f"data/persian_wikipedia_{MAX_DOCUMENTS}"
)
QDRANT_PATH = Path("data/qdrant_db")


# Preprocessing settings
MIN_CHARS = 200

# Embedding settings
MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_DEVICE = "cpu"
MAX_SEQ_LENGTH = 512

# Vector database settings
COLLECTION_NAME = f"persian_wikipedia_{MAX_DOCUMENTS}"
VECTOR_SIZE = 384

# Indexing settings
BATCH_SIZE = 64

# Retrieval settings
TOP_K = 5

# Chunking settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200