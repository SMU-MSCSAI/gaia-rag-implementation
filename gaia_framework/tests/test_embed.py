# import json
# import unittest
# import logging
# import os
# from gaia_framework.utils.embedding_processor import EmbeddingProcessor
# from gaia_framework.utils.logger_util import reset_log_file
# from gaia_framework.utils.data_object import DataObject  # Import DataObject class

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class TestEmbeddingProcessor(unittest.TestCase):
#     log_file = "./gaia_framework/tests/test_logs/data_processing_log.txt"
    
#     @classmethod
#     def setUpClass(cls):
#         # Create the log directory if it doesn't exist
#         os.makedirs(os.path.dirname(cls.log_file), exist_ok=True)
    
#     # def setUp(self):
#     #     # Reset the log file for each test
#     #     reset_log_file(self.log_file)
        
#     # def tearDown(self):
#     #     # Clean up log file after tests
#     #     if os.path.exists(self.log_file):
#     #         os.remove(self.log_file)
    
#     # Function to remove the last extra line from the JSON string
#     def clean_json_string(self, json_str):
#         lines = json_str.strip().split("\n")
#         if lines[-1] == '-----------------------------------':
#             lines = lines[:-1]
#         return "\n".join(lines)

#     def test_embed_text_huggingface(self):
#         """
#         Test embedding text using a Hugging Face model.
#         """
#         try:
#             logger.info("Running HuggingFace EmbeddingProcessor test.")
#             # Create an instance of EmbeddingProcessor with the HuggingFace model
#             processor = EmbeddingProcessor(model_name="sentence-transformers/all-MiniLM-L6-v2")
#             data_object = DataObject(
#                 id="1",
#                 domain="example.com",
#                 docsSource="source",
#                 textData="This is a test sentence."
#             )
#             # with open("./test_logs/data_processing_log.txt", "r") as file:
#             #     json_str = file.read()
#             #     clean_json_str = self.clean_json_string(json_str)

#             # Convert the cleaned JSON string to a dictionary
#             # data = json.loads(clean_json_str)
            
#             # data_object_dict = data["data"]
#             # print(data_object_dict)
#             # data_object = DataObject.from_dict(data_object_dict)
            
#             # Embed the text using the processor
#             embedding = processor.embed_text(data_object, self.log_file)

#             # Verify the log file content
#             with open(self.log_file, "r") as f:
#                 log_content = f.read()
#                 self.assertIn('"step": "Input Text"', log_content)
#                 self.assertIn('"step": "Hugging Face Embeddings"', log_content)
#                 # self.assertIn('"embedding": [0.4073353707790375, 0.280042827129364, "..."]', log_content)

#             print("\n-----------------------------------")
#             # Check if the shape of the embedding is correct
#             self.assertEqual(embedding.shape[0], 384)
#             self.assertEqual(data_object.embedding[:2], embedding.tolist()[:2])
#             logger.info("HuggingFace EmbeddingProcessor test passed.")
#             print("-----------------------------------\n")
#         except Exception as e:
#             print("\n-----------------------------------")
#             logger.error(f"Error in HuggingFace EmbeddingProcessor test: {e}")
#             self.fail(f"HuggingFace EmbeddingProcessor test failed: {e}")

#     def test_embed_text_openai(self):
#         """
#         Test embedding text using an OpenAI model.
#         """
#         try:
#             logger.info("Running OpenAI EmbeddingProcessor test.")
#             # Create an instance of EmbeddingProcessor with the OpenAI model
#             processor = EmbeddingProcessor(model_name="text-embedding-ada-002")
#             data_object = DataObject(
#                 id="1",
#                 domain="example.com",
#                 docsSource="source",
#                 textData="This is a test sentence."
#             )
#             # Embed the text using the processor
#             embedding = processor.embed_text(data_object, self.log_file)

#             # Verify the log file content
#             with open(self.log_file, "r") as f:
#                 log_content = f.read()
#                 self.assertIn('"step": "Input Text"', log_content)
#                 self.assertIn('"step": "OpenAI Embeddings"', log_content)
#                 # self.assertIn('"embedding": [0.007497059647858143, 0.01585850964486599, "..."]', log_content) # doesn't work currently

#             print("\n-----------------------------------")
#             # Check if the embedding is a list (as OpenAI returns lists)
#             self.assertIsInstance(embedding, list)
#             self.assertEqual(data_object.embedding[:2], embedding[:2])
#             logger.info("OpenAI EmbeddingProcessor test passed.")
#             print("-----------------------------------\n")
#         except Exception as e:
#             print("\n-----------------------------------")
#             logger.error(f"Error in OpenAI EmbeddingProcessor test: {e}")
#             self.fail(f"OpenAI EmbeddingProcessor test failed: {e}")
            
#     def test_unsupported_model(self):
#         """
#         Test handling of an unsupported model.
#         """
#         try:
#             logger.info("Running unsupported model test.")
#             # Attempt to create an instance of EmbeddingProcessor with an unsupported model
#             with self.assertRaises(ValueError):
#                 EmbeddingProcessor(model_name="unsupported-model")
#             print("\n-----------------------------------")
#             logger.info("Unsupported model test passed.")
#             print("-----------------------------------\n")
#         except Exception as e:
#             print("\n-----------------------------------")
#             logger.error(f"Error in unsupported model test: {e}")
#             self.fail(f"Unsupported model test failed: {e}")

# if __name__ == "__main__":
#     unittest.main()
