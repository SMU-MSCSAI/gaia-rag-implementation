import json
import logging
import os
import sys

from gaia_framework.agents.agent2.vector_db_agent import VectorDatabase
from gaia_framework.utils.chunker import TextChunker
from gaia_framework.utils.embedding_processor import EmbeddingProcessor
from gaia_framework.utils.data_object import DataObject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_and_embed_text(text_data, chunk_size, chunk_overlap):
    """
    Chunk and embed the text data.

    Args:
        text_data (str): The text data to process.
        chunk_size (int): The size of each text chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        list: List of embedded chunks.
        list: List of chunks.
    """
    chunker = TextChunker(chunk_size, chunk_overlap)
    chunks = chunker.chunk_text(text_data)
    embedder = EmbeddingProcessor()
    embeddings = [embedder.embed_text(chunk) for chunk in chunks]
    return embeddings, chunks

def initialize_vector_db(dimension, db_type, embeddings):
    """
    Initialize the vector database and add embeddings.

    Args:
        dimension (int): The dimension of the embeddings.
        db_type (str): The type of vector database.
        embeddings (list): The embeddings to add.

    Returns:
        VectorDatabase: The initialized vector database.
    """
    vector_db = VectorDatabase(dimension, db_type)
    data_object = DataObject(id="vector_db", domain="example_domain", docsSource="example_source")
    vector_db.add_embeddings(data_object, embeddings)
    return vector_db

def persist_vector_store(vector_db, path, data_object):
    """
    Persist the vector store to the local disk.

    Args:
        vector_db (VectorDatabase): The vector database instance.
        path (str): The path to save the database.
        data_object (DataObject): The data object for logging steps.
    """
    if not os.path.exists(path):
        logger.info("Persisting the vector store to the local disk...")
        vector_db.save_local(path, data_object)

def load_vector_store(vector_db, path, data_object):
    """
    Load the vector store from the local disk.

    Args:
        vector_db (VectorDatabase): The vector database instance.
        path (str): The path to load the database from.
        data_object (DataObject): The data object for logging steps.
    """
    vector_db.load_local(path, data_object)

def get_similarity_and_rag_text(vector_db, user_query, chunks, data_object):
    """
    Get similarity indices and reconstruct the relevant text.

    Args:
        vector_db (VectorDatabase): The vector database instance.
        user_query (str): The user query to search.
        chunks (list): The list of text chunks.
        data_object (DataObject): The data object for logging steps.

    Returns:
        dict: The similarity results.
        str: The reconstructed relevant text.
    """
    embedder = EmbeddingProcessor()
    query_embedding = embedder.embed_text(user_query).reshape(1, -1)
    similarity_results = vector_db.get_similarity_indices(query_embedding, data_object)
    relevant_chunks = [chunks[idx] for idx in similarity_results["indices"][0]]
    rag_text = ' '.join(relevant_chunks)
    return similarity_results, rag_text

def create_and_save_json_output(data_object, similarity_results, rag_text, db_type):
    """
    Create and save the JSON output to a file.

    Args:
        data_object (DataObject): The data object to update with results.
        similarity_results (dict): The similarity results.
        rag_text (str): The relevant text.
        db_type (str): The type of vector database.
    """
    data_object.ragText = rag_text
    data_object.vectorDB = db_type
    data_object.similarityResults = similarity_results

    json_file = "output.json"
    with open(json_file, 'w') as file:
        json.dump(data_object.to_dict(), file)
    logger.info("JSON output saved to output.json")

def main(db_type='faiss', chunk_size=512, chunk_overlap=50):
    """
    Main function to run the vector database operations.

    Args:
        db_type (str): The type of vector database. Default is 'faiss'.
        chunk_size (int): The size of each text chunk. Default is 512.
        chunk_overlap (int): The overlap between chunks. Default is 50.
    """
    try:
        text_data = "..."  # Your text data

        # Chunk and embed the text data
        embeddings, chunks = chunk_and_embed_text(text_data, chunk_size, chunk_overlap)
        dimension = embeddings[0].shape[0]

        # Initialize the vector database and add embeddings
        data_object = DataObject(id="vector_db", domain="example_domain", docsSource="example_source")
        vector_db = initialize_vector_db(dimension, db_type, embeddings)
        logger.info("Vectorizing and storing the chunks...")

        # Persist the vector store to disk if it doesn't already exist
        persist_vector_store(vector_db, 'doc_index_db', data_object)

        # Load the vector store from the local disk
        load_vector_store(vector_db, 'doc_index_db', data_object)

        user_query = "Sample user query"

        # Get similarity indices and reconstruct relevant text
        similarity_results, rag_text = get_similarity_and_rag_text(vector_db, user_query, chunks, data_object)
        logger.info("Similarity Results: %s", json.dumps(similarity_results, indent=2))

        # Create and save the JSON output
        create_and_save_json_output(data_object, similarity_results, rag_text, db_type)

        logger.info("Process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    db_type = sys.argv[1] if len(sys.argv) > 1 else 'faiss'
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    chunk_overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    main(db_type, chunk_size, chunk_overlap)
