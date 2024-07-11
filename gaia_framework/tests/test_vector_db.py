# import logging
# import unittest
# import numpy as np
# from gaia_framework.agents.agent2.vector_db_agent import VectorDatabase

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class TestVectorDatabase(unittest.TestCase):
#     def test_add_and_search_embeddings(self):
#         try:
#             logger.info("Running VectorDatabase test.")
#             dimension = 384
#             vector_db = VectorDatabase(dimension)

#             # Generate a random embedding
#             embedding = np.random.rand(1, dimension).astype(np.float32)
#             vector_db.add_embeddings(embedding)

#             # Generate a random query embedding
#             query_embedding = np.random.rand(1, dimension).astype(np.float32)
#             distances, indices = vector_db.search(query_embedding)

#             # Check the results
#             self.assertEqual(len(distances), 1)
#             self.assertEqual(len(indices), 1)
#             logger.info("VectorDatabase test passed.")
#         except Exception as e:
#             logger.error(f"Error in VectorDatabase test: {e}")
#             self.fail(f"VectorDatabase test failed: {e}")

# if __name__ == "__main__":
#     unittest.main()
    #   print("-----------------------------------")
    #   print("-----------------------------------\n")
