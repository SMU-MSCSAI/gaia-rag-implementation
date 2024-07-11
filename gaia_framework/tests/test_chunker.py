import unittest
import logging
from gaia_framework.utils.chunker import TextChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTextChunker(unittest.TestCase):
    def test_chunk_text(self):
        try:
            logging.info("Running TextChunker test.")
            text = "This is a simple text that needs to be chunked into smaller pieces."

            # Test with chunk_size=20 and chunk_overlap=5
            chunker = TextChunker(chunk_size=20, chunk_overlap=5, separator=" ")
            chunks = chunker.chunk_text(text)
            logger.info(f"Testing the number of chunks: {len(chunks)} with an overlap of 5")
            expected_chunks = [
                "This is a simple",
                "e text that needs",
                "ds to be chunked",
                "d into smaller",
                "pieces."
            ]
            self.assertEqual(chunks, expected_chunks)
            logger.info(f"TextChunker with overlap of 5: {chunks} is passed")

            # Test with chunk_size=20 and chunk_overlap=10
            chunker = TextChunker(chunk_size=20, chunk_overlap=10, separator=" ")
            chunks = chunker.chunk_text(text)
            logger.info(f"Testing the number of chunks with an overlap of 10")
           
            self.assertEqual(len(chunks), 7)
            logger.info(f"TextChunker with overlap of 10: {chunks} is passed")

            # Test with chunk_size larger than text length
            chunker = TextChunker(chunk_size=100, chunk_overlap=10, separator=" ")
            chunks = chunker.chunk_text(text)
            logger.info(f"Testing the number of chunks: {len(chunks)} with chunk_size larger than text length")
            expected_chunks = [text]
            self.assertEqual(chunks, expected_chunks)
            logger.info(f"TextChunker with chunk_size larger than text length: {chunks} is passed")

            print("\n-----------------------------------")
            logger.info("TextChunker test passed.")
            print("Finished TextChunker Test")
            print("-----------------------------------\n")
        except Exception as e:
            print("\n-----------------------------------")
            logger.error(f"Error in TextChunker test: {e}")
            self.fail(f"TextChunker test failed: {e}")

if __name__ == "__main__":
    unittest.main()
