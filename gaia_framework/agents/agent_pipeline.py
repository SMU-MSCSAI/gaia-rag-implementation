import os
import numpy as np
from gaia_framework.utils.chunker import TextChunker
from gaia_framework.utils.data_object import DataObject
from gaia_framework.utils.embedding_processor import EmbeddingProcessor
from gaia_framework.utils.logger_util import log_dataobject_step, reset_log_file

class Pipeline:
    def __init__(self, embedding_model_name, dimension, 
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
        self.dimension = dimension
        self.db_type = db_type
        self.index_type = index_type
        self.log_file = log_file
        self.embedding_model_name = embedding_model_name
        self.data_object = data_object
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = TextChunker(self.chunk_size, self.chunk_overlap, separator=",")
        self.embedder = EmbeddingProcessor(self.embedding_model_name)
        
        @classmethod
        def tearDown(self):
            # Clean up log file after tests
            if os.path.exists(self.log_file):
                os.remove(self.log_file)

    def process_data(self):
        """
        Process the data by chunking and embedding the text data.

        Args:
            data (list): The list of text data to process.
        """
        # get the chunks and the data object
        chunks, data_object = self.chunker.chunk_text(self.data_object, self.log_file)
        #embed the chunks
        embeddings, data_object = self.embedder.embed_text(self.data_object, self.log_file)
        self.data_object = data_object
        return embeddings, data_object
        

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
    
    pipeline = Pipeline(embedding_model_name="text-embedding-ada-002",
                        dimension=384, 
                        data_object=data_object,
                        db_type='faiss', 
                        index_type='FlatL2', 
                        chunk_size=15,
                        chunk_overlap=5,
                        log_file=log_file)

    embeddings, data_object = pipeline.process_data()
    print(embeddings)
    print(data_object)