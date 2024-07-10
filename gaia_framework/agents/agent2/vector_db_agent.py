import faiss
import numpy as np

class VectorDatabase:
    def __init__(self, dimension, db_type='faiss'):
        self.dimension = dimension
        self.db_type = db_type
        self.index = self._create_index()
        self.data = []

    def _create_index(self):
        if self.db_type == 'faiss':
            return faiss.IndexFlatL2(self.dimension)
        elif self.db_type == 'other_db':
            # Placeholder for another vector database initialization
            # Implement other vector database initialization here
            pass
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def add_embeddings(self, embeddings):
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        self.index.add(embeddings)
    
    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices