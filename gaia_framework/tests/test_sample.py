import unittest
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSample(unittest.TestCase):
    def test_example(self):
        try:
            logger.info("Running sample test.")
            self.assertEqual(1 + 1, 2)
            logger.info("Sample test passed.")
        except Exception as e:
            logger.error(f"Error in Sample test: {e}")
            self.fail(f"Sample test failed: {e}")

if __name__ == "__main__":
    unittest.main()
