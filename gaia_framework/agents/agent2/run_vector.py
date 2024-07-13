import json
import logging
import os
from gaia_framework.agents.agent2.vector_db_agent import VectorDatabase
from gaia_framework.utils.chunker import TextChunker
from gaia_framework.utils.embedding_processor import EmbeddingProcessor

def main(db_type='faiss', chunk_size=512, chunk_overlap=50):
    """
    Main function to run the vector database operations.

    Args:
        db_type (str): The type of vector database. Default is 'faiss'.
        chunk_size (int): The size of each text chunk. Default is 512.
        chunk_overlap (int): The overlap between chunks. Default is 50.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        embedder = EmbeddingProcessor()
        text_data = "..."  # your text data
        chunker = TextChunker(chunk_size, chunk_overlap)
        chunks = chunker.chunk_text(text_data)

        # Embed the text chunks
        embeddings = [embedder.embed_text(chunk) for chunk in chunks]
        dimension = embeddings[0].shape[0]

        # Initialize the vector database and add embeddings
        vector_db = VectorDatabase(dimension, db_type)
        vector_db.add_embeddings(embeddings)

        print("Vectorizing and storing the chunks...\n")

        # Persist the vector store to disk if it doesn't already exist
        if not os.path.exists('doc_index_db'):
            print("Persisting the vector store to the local disk...\n")
            vector_db.save_local('doc_index_db')

        # Load the vector store from the local disk
        vector_db.load_local('doc_index_db')

        user_query = "Sample user query"
        query_embedding = embedder.embed_text(user_query).reshape(1, -1)

        # Get similarity indices
        similarity_results = vector_db.get_similarity_indices(query_embedding)
        print("Similarity Results:", json.dumps(similarity_results, indent=2))

        # Extract relevant chunks based on similarity indices
        relevant_chunks = [chunks[idx] for idx in similarity_results["indices"][0]]
        rag_text = ' '.join(relevant_chunks)

        # Save the relevant chunks text to a file
        rag_text_file = "rag_text.txt"
        with open(rag_text_file, 'w') as file:
            file.write(rag_text)

        # Create the JSON output
        updated_json = {
            "id": "<project name>",
            "domain": "<legal academic ...>",
            "docsSource": "<list of sources>",
            "queries": "<list of example queries>",
            "textData": "<file reference>",
            "embedding": "sentence-transformers/all-MiniLM-L6-v2",
            "vectorDB": db_type,
            "ragText": rag_text_file,
            "similarityResults": similarity_results
        }

        # Save the JSON output to a file
        json_file = "output.json"
        with open(json_file, 'w') as file:
            json.dump(updated_json, file)

        logger.info("Process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    import sys
    db_type = sys.argv[1] if len(sys.argv) > 1 else 'faiss'
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    chunk_overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    main(db_type, chunk_size, chunk_overlap)
