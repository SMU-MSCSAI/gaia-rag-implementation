import json
import logging
from gaia_framework.agents.agent2.vector_db_agent import VectorDatabase
from gaia_framework.utils.chunker import TextChunker
from gaia_framework.utils.embedding_processor import EmbeddingProcessor


def main(db_type="faiss", chunk_size=512, chunk_overlap=50):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        embedder = EmbeddingProcessor()
        text_data = "..."  # your text data
        chunker = TextChunker(chunk_size, chunk_overlap)
        chunks = chunker.chunk_text(text_data)

        embeddings = [embedder.embed_text(chunk) for chunk in chunks]
        dimension = embeddings[0].shape[0]
        vector_db = VectorDatabase(dimension, db_type)
        vector_db.add_embeddings(embeddings)

        user_query = "Sample user query"
        query_embedding = embedder.embed_text(user_query).reshape(1, -1)
        distances, indices = vector_db.search(query_embedding)

        relevant_chunks = [chunks[idx] for idx in indices[0]]
        rag_text = " ".join(relevant_chunks)

        rag_text_file = "rag_text.txt"
        with open(rag_text_file, "w") as file:
            file.write(rag_text)

        updated_json = {
            "id": "<project name>",
            "domain": "<legal academic ...>",
            "docsSource": "<list of sources>",
            "queries": "<list of example queries>",
            "textData": "<file reference>",
            "embedding": "sentence-transformers/all-MiniLM-L6-v2",
            "vectorDB": db_type,
            "ragText": rag_text_file,
        }

        json_file = "output.json"
        with open(json_file, "w") as file:
            json.dump(updated_json, file)

        logger.info("Process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
