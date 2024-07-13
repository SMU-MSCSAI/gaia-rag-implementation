# import unittest
# import logging
# from gaia_framework.utils.embedding_processor import EmbeddingProcessor

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class TestEmbeddingProcessor(unittest.TestCase):
#     def test_embed_text_huggingface(self):
#         """
#         Test embedding text using a Hugging Face model.
#         """
#         try:
#             logger.info("Running HuggingFace EmbeddingProcessor test.")
#             # Create an instance of EmbeddingProcessor with the HuggingFace model
#             processor = EmbeddingProcessor(model_name="sentence-transformers/all-MiniLM-L6-v2")
#             text = "This is a test sentence."
#             # Embed the text using the processor
#             embedding = processor.embed_text(text)

#             print("\n-----------------------------------")
#             # Check if the shape of the embedding is correct
#             self.assertEqual(embedding.shape[0], 384)
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
#             processor = EmbeddingProcessor(model_name="text-embedding-3-small")
#             text = "This is a test sentence."
#             # Embed the text using the processor
#             embedding = processor.embed_text(text)
#             print("\n-----------------------------------")
#             # Check if the embedding is a list (as OpenAI returns lists)
#             self.assertIsInstance(embedding, list)
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

