import logging
import os
import numpy as np
from gaia_framework.agents.agent2.vector_db_agent import VectorDatabase
from gaia_framework.utils.chunker import TextChunker
from gaia_framework.utils.data_object import DataObject
from gaia_framework.utils.embedding_processor import EmbeddingProcessor
from gaia_framework.utils.logger_util import reset_log_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding dimensions for each model
embedding_dimensions = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "bert-base-uncased": 768,
    "roberta-base": 768,
    "text-embedding-ada-002": 1536,
    "text-embedding-babbage-001": 2048,
    # Add more models and their dimensions as needed
}

class Pipeline:
    def __init__(self, embedding_model_name, 
                data_object: DataObject, db_type='faiss', 
                index_type='FlatL2', 
                chunk_size=512, 
                chunk_overlap=50,
                log_file="data_processing_log.txt"):
        """
            Usage of the Pipeline class.
            1. Get the data as text to be used as the context. 
            2. Embed the text data in chunks.
            3. Initialize the vector database and add the embeddings.
            4. Take a query and embed it.
            5. Search for the top k most similar embeddings from the vector database.
            6. Return the results to be used as the context for the language model.
            7. Run the language model with the context and query.
            8. Get the response from the language model.

        Args:
            dimension (_type_): dimension of the embeddings.
            db_type (str, optional): database type. Defaults to 'faiss'.
            index_type (str, optional): index type. Defaults to 'FlatL2'.
            log_file (str, optional): log file. Defaults to "data_processing_log.txt".
        """
        # Retrieve the correct dimension for the selected model
        self.dimension = embedding_dimensions.get(embedding_model_name)
        if self.dimension is None:
            raise ValueError(f"Embedding dimension for model '{embedding_model_name}' not found.")

        self.db_type = db_type
        self.index_type = index_type
        self.log_file = log_file
        self.embedding_model_name = embedding_model_name
        self.data_object = data_object
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.chunker = TextChunker(self.chunk_size, self.chunk_overlap, separator=",")
        self.embedder = EmbeddingProcessor(self.embedding_model_name)
        # instantiate the class, and create the index (data structure sufficient to store the embeddings)
        self.vector_db = VectorDatabase(self.dimension, self.db_type, self.index_type) 
        
        # restart the log file after each run
        reset_log_file(self.log_file)
        # clean up the logging file after each run
        @classmethod
        def tearDown(self):
            # Clean up log file after tests
            if os.path.exists(self.log_file):
                os.remove(self.log_file)

    def process_data_chunk(self):
        """
        Process the data by chunking and embedding the text data.

        Args:
            data (list): The list of text data to process.
        """
        # get the chunks and the data object
        chunks, data_object = self.chunker.chunk_text(self.data_object, self.log_file)
        return chunks, data_object
    
    def process_data_embed(self):
        #embed the chunks
        embeddings, data_object = self.embedder.embed_text(self.data_object, self.log_file)
        self.data_object = data_object
        return embeddings, data_object
        
    def add_embeddings(self, embeddings):
        """
        Add the embeddings to the vector database.

        Args:
            embeddings (list): The list of embeddings to add.
        """
        self.vector_db.add_embeddings(self.data_object, embeddings, self.log_file)
        
    def save_local(self, path):
        """
        Persist the vector store to the local disk.

        Args:
            path (str): The path to save the database.
        """
        if not os.path.exists(path):
            self.vector_db.save_local(path, self.data_object, self.log_file)
        else: 
            logger.warn(f"Path {path} already exists. Please provide a db name to create another index db locally.")
            
    def load_local(self, path):
        """
        Load the vector store from the local disk.

        Args:
            path (str): The path to load the database from.
        """
        if os.path.exists(path):
            self.vector_db.load_local(path, self.data_object, self.log_file)
            return True
        else:
            logger.warning(f"Path {path} does not exist. Please provide a valid path to load the index db or save the db first.")
            return False
        
    # def search_embeddings(self, query_embedding, k=5):
    #     """
    #     Search the vector database for the top k most similar embeddings.

    #     Args:
    #         query_embedding (np.ndarray): The query embedding to search for.
    #         k (int): The number of top similar embeddings to return. Default is 5.

    #     Returns:
    #         dict: A dictionary containing the distances and indices of the top k similar embeddings.
    #     """
    #     return self.vector_db.get_similarity_indices(query_embedding, self.data_object, k)
if __name__ == "__main__":
    data = "This is a test sentence about domestic animals, Here I come with another test sentence about the cats."
    data_object = DataObject(
        id="test_id",
        domain="test_domain",
        textData=data,
        docsSource="test_docsSource",
        queries=["what's this test about?"],
        chunks=[],
        embedding=None,
        vectorDB=None,
        ragText=None,
        llmResult=None
    )
    
    log_file = "./data/data_processing_log.txt"
    db_path = "./data/doc_index_db"
    
    pipeline = Pipeline(embedding_model_name="text-embedding-ada-002",
                        data_object=data_object,
                        db_type='faiss', 
                        index_type='FlatL2', 
                        chunk_size=15,
                        chunk_overlap=5,
                        log_file=log_file)
    # 1. Chunck and embed the text 
    chunks, data_object = pipeline.process_data_chunk()
    
    # 2. If the db is already saved, load it, otherwise add the embeddings and save it
    logger.info("Trying to Load the vector store from the local disk...")
    if not pipeline.load_local(db_path):
        # 2.1 Embed the chunks
        embeddings, data_object = pipeline.process_data_embed()
        #2.2 Add embeddings to the vector database
        pipeline.add_embeddings(embeddings)
        #2.3 Persist the vector store to the local disk or load if exist
        pipeline.save_local(db_path)
    
    print("Pipeline completed successfully, check the log file for more details.")