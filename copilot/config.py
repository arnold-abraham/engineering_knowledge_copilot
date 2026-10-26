import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4

UPLOAD_DIR = "uploaded_docs"
SUPPORTED_EXTENSIONS = ["pdf", "py", "cpp", "c", "txt", "md"]
