import logging
import unittest
import numpy as np
import os
import json

from gaia_framework.agents.agent2.vector_db_agent import VectorDatabase
from gaia_framework.utils.logger_util import reset_log_file, log_dataobject_step
from gaia_framework.utils.data_object import DataObject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestVectorDatabase(unittest.TestCase):
    log_file = "./gaia_framework/tests/test_logs/data_processing_log.txt"
    
    @classmethod
    def setUpClass(cls):
        # Create the log directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.log_file), exist_ok=True)
        
    def setUp(self, db_type='faiss'):
        # Reset the log file for each test
        reset_log_file(self.log_file)
        self.dimension = 384
        self.db_type = db_type
        self.vector_db = VectorDatabase(self.dimension, self.db_type)
        self.embedding = np.random.rand(1, self.dimension).astype(np.float32)
        self.query_embedding = np.random.rand(1, self.dimension).astype(np.float32)
        self.test_path = "./gaia_framework/tests/test_db/"
        self.data_object = DataObject(id="test", domain="unit_test", docsSource="test_source")
      
    # def tearDown(self):
    #     if os.path.exists(self.test_path):
    #         for file in os.listdir(self.test_path):
    #             os.remove(os.path.join(self.test_path, file))
    #         os.rmdir(self.test_path)
        # # Clean up log file after tests
        # if os.path.exists(self.log_file):
        #     os.remove(self.log_file)

    def read_log_file(self):
        with open(self.log_file, 'r') as file:
            log_data = file.readlines()
        return log_data

    def test_add_and_search_embeddings(self):
        try:
            logger.info(f"Running add and search embeddings test with {self.db_type}.")
            self.vector_db.add_embeddings(self.data_object, self.embedding, self.log_file)
            distances, indices = self.vector_db.search(self.data_object, self.query_embedding, log_file=self.log_file)

            # Read log file and verify
            log_data = self.read_log_file()
            self.assertIn("Embeddings Added and Indexed", log_data[-1])
            self.assertIn("Embeddings Searched", log_data[-1])

            self.assertEqual(distances.shape[0], 1)
            self.assertEqual(indices.shape[0], 1)
            logger.info(f"Add and search embeddings test with {self.db_type} passed.")
        except Exception as e:
            logger.error(f"Error in add and search embeddings test with {self.db_type}: {e}")
            self.fail(f"Add and search embeddings test with {self.db_type} failed: {e}")

    def test_save_and_load_local(self):
        try:
            logger.info(f"Running save and load local test with {self.db_type}.")
            self.vector_db.add_embeddings(self.data_object, self.embedding, self.log_file)
            self.vector_db.save_local(self.test_path, self.data_object, self.log_file)

            new_vector_db = VectorDatabase(self.dimension, self.db_type)
            new_vector_db.load_local(self.test_path, self.data_object, self.log_file)

            distances, indices = new_vector_db.search(self.data_object, self.query_embedding, log_file=self.log_file)

            # Read log file and verify
            log_data = self.read_log_file()
            self.assertIn("Embeddings Saved Locally", log_data[-1])
            self.assertIn("Embeddings Loaded Locally", log_data[-1])

            self.assertEqual(distances.shape[0], 1)
            self.assertEqual(indices.shape[0], 1)
            logger.info(f"Save and load local test with {self.db_type} passed.")
        except Exception as e:
            logger.error(f"Error in save and load local test with {self.db_type}: {e}")
            self.fail(f"Save and load local test with {self.db_type} failed: {e}")

if __name__ == "__main__":
    # Run tests for different database types
    for db_type in ['faiss']:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestVectorDatabase)
        for test in suite:
            test.setUp(db_type=db_type)
        unittest.TextTestRunner().run(suite)
