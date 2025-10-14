import os
from urllib.parse import urlparse
import chromadb


def get_chroma_client():
    """
    Creates and returns a ChromaDB HttpClient from the CHROMA_HOST environment variable.
    """
    chroma_host = os.getenv("CHROMA_HOST")
    if not chroma_host:
        raise ValueError("CHROMA_HOST environment variable is not set.")

    parsed_url = urlparse(chroma_host)
    if not parsed_url.hostname or not parsed_url.port:
        raise ValueError(
            f"Invalid CHROMA_HOST URL: {chroma_host}. Expected format: http://host:port"
        )

    return chromadb.HttpClient(
        host=parsed_url.hostname,
        port=parsed_url.port,
        ssl=(parsed_url.scheme == "https"),
    )
