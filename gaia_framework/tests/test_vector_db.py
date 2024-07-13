# import logging
# import unittest
# import numpy as np
# import os
# from gaia_framework.agents.agent2.vector_db_agent import VectorDatabase

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class TestVectorDatabase(unittest.TestCase):
#     def setUp(self, db_type='faiss'):
#         self.dimension = 384
#         self.db_type = db_type
#         self.vector_db = VectorDatabase(self.dimension, self.db_type)
#         self.embedding = np.random.rand(1, self.dimension).astype(np.float32)
#         self.query_embedding = np.random.rand(1, self.dimension).astype(np.float32)
#         self.test_path = './test_db/'

#     def tearDown(self):
#         if os.path.exists(self.test_path):
#             for file in os.listdir(self.test_path):
#                 os.remove(os.path.join(self.test_path, file))
#             os.rmdir(self.test_path)

#     def test_add_and_search_embeddings(self):
#         try:
#             logger.info(f"Running add and search embeddings test with {self.db_type}.")
#             self.vector_db.add_embeddings(self.embedding)
#             distances, indices = self.vector_db.search(self.query_embedding)

#             self.assertEqual(distances.shape[0], 1)
#             self.assertEqual(indices.shape[0], 1)
#             logger.info(f"Add and search embeddings test with {self.db_type} passed.")
#         except Exception as e:
#             logger.error(f"Error in add and search embeddings test with {self.db_type}: {e}")
#             self.fail(f"Add and search embeddings test with {self.db_type} failed: {e}")

#     def test_save_and_load_local(self):
#         try:
#             logger.info(f"Running save and load local test with {self.db_type}.")
#             self.vector_db.add_embeddings(self.embedding)
#             self.vector_db.save_local(self.test_path)

#             new_vector_db = VectorDatabase(self.dimension, self.db_type)
#             new_vector_db.load_local(self.test_path)

#             distances, indices = new_vector_db.search(self.query_embedding)

#             self.assertEqual(distances.shape[0], 1)
#             self.assertEqual(indices.shape[0], 1)
#             logger.info(f"Save and load local test with {self.db_type} passed.")
#         except Exception as e:
#             logger.error(f"Error in save and load local test with {self.db_type}: {e}")
#             self.fail(f"Save and load local test with {self.db_type} failed: {e}")

# if __name__ == "__main__":
#     # Run tests for different database types
#     for db_type in ['faiss', 'other_db']:
#         suite = unittest.TestLoader().loadTestsFromTestCase(TestVectorDatabase)
#         for test in suite:
#             test.setUp(db_type=db_type)
#         unittest.TextTestRunner().run(suite)
