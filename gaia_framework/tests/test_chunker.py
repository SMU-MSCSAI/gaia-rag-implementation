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
            print(chunks)
            # Check if the number of chunks is correct
            self.assertEqual(len(chunks), 4) 
            # Check if the first chunk contains the expected text
            self.assertIn("This is a simple text", chunks[0])  
            logger.info("TextChunker test passed.")
        except Exception as e:
            logger.error(f"Error in TextChunker test: {e}")
            self.fail(f"TextChunker test failed: {e}")

if __name__ == "__main__":
    unittest.main()
