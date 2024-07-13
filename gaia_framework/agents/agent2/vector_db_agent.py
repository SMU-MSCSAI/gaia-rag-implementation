import os
import faiss
import numpy as np

class VectorDatabase:
    def __init__(self, dimension, db_type='faiss'):
        """
        Initialize the vector database.

        Args:
            dimension (int): The dimension of the embeddings.
            db_type (str): The type of vector database. Default is 'faiss'.
        """
        self.dimension = dimension
        self.db_type = db_type
        self.index = self._create_index()
        self.data = []

    def _create_index(self):
        """
        Create the index for the vector database based on the db_type.

        Returns:
            faiss.Index: The FAISS index if db_type is 'faiss'.
        """
        if self.db_type == 'faiss':
            return faiss.IndexFlatL2(self.dimension)
        elif self.db_type == 'other_db':
            # Placeholder for another vector database initialization
            # Implement other vector database initialization here
            pass
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def add_embeddings(self, embeddings):
        """
        Add embeddings to the vector database.

        Args:
            embeddings (list or np.ndarray): The embeddings to add.
        """
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings).astype(np.float32)
        self.index.add(embeddings)
        self.data.extend(embeddings)

    def search(self, query_embedding, k=5):
        """
        Search the vector database for the top k most similar embeddings.

        Args:
            query_embedding (np.ndarray): The query embedding to search for.
            k (int): The number of top similar embeddings to return. Default is 5.

        Returns:
            tuple: Distances and indices of the top k similar embeddings.
        """
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding).astype(np.float32)
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def save_local(self, path):
        """
        Save the vector database to a local directory.

        Args:
            path (str): The path to the directory where the database will be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        np.save(os.path.join(path, 'data.npy'), np.array(self.data))

    def load_local(self, path):
        """
        Load the vector database from a local directory.

        Args:
            path (str): The path to the directory where the database is saved.
        """
        self.index = faiss.read_index(os.path.join(path, 'index.faiss'))
        self.data = np.load(os.path.join(path, 'data.npy')).tolist()

    def get_similarity_indices(self, query_embedding, k=5):
        """
        Get the top k similar embeddings' indices and distances.

        Args:
            query_embedding (np.ndarray): The query embedding to search for.
            k (int): The number of top similar embeddings to return.

        Returns:
            dict: A dictionary containing the distances and indices of the top k similar embeddings.
        """
        distances, indices = self.search(query_embedding, k)
        results = {
            "distances": distances.tolist(),
            "indices": indices.tolist()
        }
        return results
