# import unittest
# import logging
# import os
# from gaia_framework.utils.chunker import TextChunker, DataObject
# from gaia_framework.utils.logger_util import reset_log_file, log_dataobject_step

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class TestTextChunker(unittest.TestCase):
#     """
#     TestTextChunker is a class for testing the functionality of the TextChunker class.
#     """
    
#     log_file = "./gaia_framework/tests/test_logs/data_processing_log.txt"
    
#     @classmethod
#     def setUpClass(cls):
#         # Create the log directory if it doesn't exist
#         os.makedirs(os.path.dirname(cls.log_file), exist_ok=True)
    
#     def setUp(self):
#         # Reset the log file for each test
#         reset_log_file(self.log_file)

#     def tearDown(self):
#         # Clean up log file after tests
#         if os.path.exists(self.log_file):
#             os.remove(self.log_file)

#     def test_chunk_text(self):
#         """
#         Test the chunk_text method of the TextChunker class.
#         """
#         try:
#             logging.info("Running TextChunker test.")
#             text = "This is a simple text that needs to be chunked into smaller pieces."
#             data_object = DataObject(
#                 id="1",
#                 domain="example.com",
#                 docsSource="source",
#                 textData=text
#             )

#             # Test with chunk_size=20 and chunk_overlap=5
#             chunker = TextChunker(chunk_size=20, chunk_overlap=5, separator=" ")
#             chunks = chunker.chunk_text(data_object, self.log_file)
#             logger.info(f"Testing the number of chunks: {len(chunks)} with an overlap of 5")
#             expected_chunks = [
#                 'This is a simple', 
#                 'e text that needs', 
#                 'ds to be chunked', 
#                 'd into smaller', 
#                 'pieces.'
#             ]
#             self.assertEqual(chunks, expected_chunks)
#             self.assertEqual(data_object.chunks, expected_chunks)
#             logger.info(f"TextChunker with overlap of 5: with {len(chunks)} chunks is PASSED!!")

#             # Verify the log file content after chunking
#             with open(self.log_file, "r") as f:
#                 log_content = f.read()
#                 self.assertIn("Step: After Chunking", log_content)
#                 for chunk in expected_chunks:
#                     self.assertIn(chunk, log_content)

#             # Test with chunk_size=20 and chunk_overlap=10
#             chunker = TextChunker(chunk_size=20, chunk_overlap=10, separator=" ")
#             chunks = chunker.chunk_text(data_object, self.log_file)
#             logger.info(f"Testing the number of chunks with an overlap of 10")
#             expected_chunks = [
#                 'This is a simple', 
#                 'simple text that', 
#                 't that needs to be', 
#                 'ds to be chunked', 
#                 'hunked into smaller', 
#                 'o smaller pieces.', 
#                 'pieces.'
#             ]
#             self.assertEqual(chunks, expected_chunks)
#             self.assertEqual(data_object.chunks, expected_chunks)
#             logger.info(f"TextChunker with overlap of 10: {chunks} is passed")

#             # Verify the log file content after chunking
#             with open(self.log_file, "r") as f:
#                 log_content = f.read()
#                 self.assertIn("Step: After Chunking", log_content)
#                 for chunk in expected_chunks:
#                     self.assertIn(chunk, log_content)

#             # Test with chunk_size larger than text length
#             chunker = TextChunker(chunk_size=100, chunk_overlap=10, separator=" ")
#             chunks = chunker.chunk_text(data_object, self.log_file)
#             logger.info(f"Testing the number of chunks: {len(chunks)} with chunk_size larger than text length")
#             expected_chunks = [text]
#             self.assertEqual(chunks, expected_chunks)
#             self.assertEqual(data_object.chunks, expected_chunks)
#             logger.info(f"TextChunker with chunk_size larger than text length: {chunks} is passed")

#             # Log the final state after tests
#             log_dataobject_step(data_object, "Final State after tests", self.log_file)

#             # Verify the log file content after final state
#             with open(self.log_file, "r") as f:
#                 log_content = f.read()
#                 self.assertIn("Step: Final State after tests", log_content)
#                 self.assertIn(text, log_content)

#             print("\n-----------------------------------")
#             logger.info("TextChunker test passed.")
#             print("Finished TextChunker Test")
#             print("-----------------------------------\n")
#         except Exception as e:
#             print("\n-----------------------------------")
#             logger.error(f"Error in TextChunker test: {e}")
#             self.fail(f"TextChunker test failed: {e}")

    # def test_reset_log_file(self):
    #     """
    #     Test the reset_log_file function.
    #     """
    #     try:
    #         logging.info("Running reset_log_file test.")

    #         # Write some initial content to the log file
    #         with open(self.log_file, "a") as f:
    #             f.write("Initial content\n")

    #         # Reset the log file
    #         reset_log_file(self.log_file)

    #         # Verify the log file content
    #         with open(self.log_file, "r") as f:
    #             log_content = f.read()
    #             self.assertNotIn("Initial content", log_content)
    #             self.assertIn("Data Processing Log", log_content)

    #         logger.info("reset_log_file test passed.")
    #     except Exception as e:
    #         print("\n-----------------------------------")
    #         logger.error(f"Error in reset_log_file test: {e}")
    #         self.fail(f"reset_log_file test failed: {e}")

# if __name__ == "__main__":
#     unittest.main()
