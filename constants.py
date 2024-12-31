import os
from chromadb.config import Settings

PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'db')

CHROMA_SETTINGS = Settings(
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)
