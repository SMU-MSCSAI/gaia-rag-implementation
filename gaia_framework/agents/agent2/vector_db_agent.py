import logging
import os
import faiss
import numpy as np

from gaia_framework.utils.data_object import DataObject
from gaia_framework.utils.logger_util import log_dataobject_step

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, dimension, db_type='faiss', index_type='FlatL2'):
        """
        Initialize the vector database.

        Args:
            dimension (int): The dimension of the embeddings.
            db_type (str): The type of vector database. Default is 'faiss'.
            index_type (str): The type of FAISS index. Default is 'FlatL2'.
        """
        self.dimension = dimension
        self.db_type = db_type
        self.index_type = index_type
        self.index = self._create_index()
        self.data = []

    def _create_index(self):
        """
        Create the index for the vector database based on the db_type.

        Returns:
            faiss.Index: The FAISS index if db_type is 'faiss'.
        """
        if self.db_type == 'faiss':
            logger.info("Creating FAISS db index.")
            if self.index_type == 'FlatL2':
                logger.info("Create FAISS using FlatL2 index.")
                return faiss.IndexFlatL2(self.dimension)
            elif self.index_type == 'IVFFlat':
                logger.info("Create FAISS using IVFFlat index.")
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                index.nprobe = 10  # set the number of probes for IVFFlat
                return index
            else:
                logger.error(f"Unsupported FAISS index type: {self.index_type}")
                raise ValueError(f"Unsupported FAISS index type: {self.index_type}")
        elif self.db_type == 'other_db':
            logger.info("Creating other vector database index.")
            # Placeholder for another vector database initialization
            # Implement other vector database initialization here
            pass
        else:
            logger.error(f"Unsupported database type: {self.db_type}")
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def add_embeddings(self, data_object: DataObject, embeddings, log_file: str = "data_processing_log.txt"):
        """
        Add embeddings to the vector database.

        Args:
            data_object (DataObject): The data object containing the text to generate embeddings for.
            embeddings (list or np.ndarray): The embeddings to add.
            log_file (str): The file to log processing steps.
        """
        try:
            log_dataobject_step(data_object, "Input Text to embedding indexing", log_file)
            logger.info("Adding embeddings to the vector database.")
            
            if isinstance(embeddings, (tuple, list)):
                embeddings = np.array([np.array(embedding).astype(np.float32) for embedding in embeddings])
            
            if embeddings.ndim == 1:
                embeddings = np.expand_dims(embeddings, axis=0)
            
            if embeddings.shape[1] != self.dimension:
                raise ValueError(f"Embedding dimension does not match the index dimension. Expected {self.dimension}, got {embeddings.shape[1]}.")
            
            self.index.add(embeddings)
            self.data.extend(embeddings.tolist())
            data_object.vectorDB = self.db_type
            data_object.embeddingAdded = True
            log_dataobject_step(data_object, "After Embeddings Added and Indexed", log_file)
        except (ValueError, TypeError) as e:
            logger.error(f"Error adding embeddings: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def search(self, data_object: DataObject, query_embedding, k=5, log_file: str = "data_processing_log.txt"):
        """
        Search the vector database for the top k most similar embeddings.

        Args:
            data_object (DataObject): The data object containing the text to generate embeddings for.
            query_embedding (np.ndarray): The query embedding to search for.
            k (int): The number of top similar embeddings to return. Default is 5.

        Returns:
            tuple: Distances and indices of the top k similar embeddings.
        """
        query_embedding = query_embedding[0]
        try:
            logger.info("Searching embeddings in the vector database.")
            
            # Ensure query_embedding is a numpy array
            if isinstance(query_embedding, (tuple, list)):
                query_embedding = np.array([np.array(embedding).astype(np.float32) for embedding in query_embedding])
                
            elif isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding).astype(np.float32)
                
            if query_embedding.ndim == 1:
                query_embedding = np.expand_dims(query_embedding, axis=0)
                
            if query_embedding.shape[1] != self.dimension:
                raise ValueError("Query embedding dimension does not match the index dimension.")
                
            logger.info(f"Query embedding shape: {query_embedding.shape}")
            distances, indices = self.index.search(query_embedding, k)
            return distances, indices
        except (ValueError, TypeError) as e:
            logger.error(f"Error searching embeddings: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def save_local(self, path, data_object: DataObject, log_file: str = "data_processing_log.txt"):
        """
        Save the vector database to a local directory.

        Args:
            path (str): The path to the directory where the database will be saved.
            data_object (DataObject): The data object containing the text to generate embeddings for.
            log_file (str): The file to log processing steps.
        """
        try:
            log_dataobject_step(data_object, "Input Text to save local", log_file)
            logger.info(f"Saving embeddings locally to {path}")
            os.makedirs(path, exist_ok=True)
            faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
            np.save(os.path.join(path, 'data.npy'), np.array(self.data))
            logger.info(f"Embeddings saved locally to {path}")
            data_object.vectorDBPersisted = True
            log_dataobject_step(data_object, "After Embeddings Saved Locally", log_file)
        except (OSError, ValueError) as e:
            logger.error(f"Error saving embeddings locally: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def load_local(self, path, data_object: DataObject, log_file: str = "data_processing_log.txt"):
        """
        Load the vector database from a local directory.

        Args:
            path (str): The path to the directory where the database is saved.
            data_object (DataObject): The data object containing the text to generate embeddings for.
            log_file (str): The file to log processing steps.
        """
        try:
            log_dataobject_step(data_object, "Input Text to load local", log_file)
            logger.info(f"Loading embeddings locally from {path}")
            if not os.path.exists(path):
                logger.error(f"Directory not found: {path}")
                raise FileNotFoundError(f"Directory not found: {path}")
            self.index = faiss.read_index(os.path.join(path, 'index.faiss'))
            self.data = np.load(os.path.join(path, 'data.npy')).tolist()
            logger.info(f"Embeddings loaded from {path}")
            data_object.vectorDB = self.db_type
            data_object.vectorDBLoaded = True
            log_dataobject_step(data_object, "After Embeddings Loaded Locally", log_file)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading embeddings locally: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def get_similarity_indices(self, query_embedding, data_object: DataObject, k=5, log_file: str = "data_processing_log.txt"):
        """
        Get the top k similar embeddings' indices and distances.

        Args:
            query_embedding (np.ndarray): The query embedding to search for.
            k (int): The number of top similar embeddings to return.
            data_object (DataObject): The data object containing the text to generate embeddings for.
            log_file (str): The file to log processing steps.

        Returns:
            dict: A dictionary containing the distances and indices of the top k similar embeddings.
        """
        try:
            log_dataobject_step(data_object, "Input Text to get similarity indices", log_file)
            logger.info("Getting similarity indices.")
            distances, indices = self.search(data_object, query_embedding, k)
            similarity_results = {
                "distances": distances.tolist(),
                "indices": indices.tolist()
            }
            logger.info(f"Similarity indices: {similarity_results}")
            data_object.vectorDB = self.db_type
            data_object.similarityIndices = similarity_results.get("indices")
            log_dataobject_step(data_object, "After Similarity Indices Retrieved", log_file)
            return similarity_results
        except (ValueError, TypeError) as e:
            logger.error(f"Error getting similarity indices: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
