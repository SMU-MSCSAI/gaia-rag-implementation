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
            chunker = TextChunker(chunk_size=5, chunk_overlap=1)
            chunks = chunker.chunk_text(text)
            # Check if the number of chunks is correct
            logger.info(f"Testing the number of chunks: {len(chunks)} with an overlap of 1")
            self.assertEqual(len(chunks), 4)
            logger.info(f"TextChunker with overlap of 1: {chunks} is passed")

            chunker = TextChunker(chunk_size=5, chunk_overlap=2)
            two_chunks = chunker.chunk_text(text)
            # Check if the number of chunks is correct
            logger.info(f"Testing the number of chunks: {len(chunks)} with an overlap of 2")
            self.assertEqual(len(two_chunks), 5)
            logger.info(f"TextChunker with overlap of 2: {chunks} is passed")
            # Check if the first chunk contains the expected text
            self.assertIn("This is a simple text", chunks[0])
            print("\n-----------------------------------")
            logger.info("TextChunker test passed.")
            print("Finished Sample Test")
            print("-----------------------------------\n")
        except Exception as e:
            print("\n-----------------------------------")
            logger.error(f"Error in TextChunker test: {e}")
            self.fail(f"TextChunker test failed: {e}")

if __name__ == "__main__":
    unittest.main()
