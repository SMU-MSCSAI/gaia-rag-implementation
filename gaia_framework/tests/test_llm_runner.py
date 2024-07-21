import os
import unittest
from gaia_framework.agents.agent3.llm_runner import LLMRunner
from gaia_framework.agents.agent3.llm_data_manager import DataObjectManager
from gaia_framework.utils.data_object import DataObject
from gaia_framework.utils.logger_util import reset_log_file

class TestLLMRunner(unittest.TestCase):
    log_file = "./gaia_framework/tests/test_logs/data_processing_log.txt"

    @classmethod
    def setUpClass(cls):
        # Create the log directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.log_file), exist_ok=True)

    def setUp(self):
        # Reset the log file for each test
        reset_log_file(self.log_file)
        
        # Initialize a sample data object
        self.data_object = DataObject(id="test", domain="unit_test", docsSource="test_source", ragText="This is a test context.")
        self.data_manager = DataObjectManager(self.data_object, self.log_file)

    def read_log_file(self):
        with open(self.log_file, 'r') as file:
            log_data = file.readlines()
        return log_data

    def test_run_query_with_local_endpoint(self):
        local_endpoint = "http://localhost:8000/llm"
        llm_runner = LLMRunner(local_endpoint=local_endpoint)

        context = self.data_manager.data_object.ragText
        query = "What are the key points from the provided text?"

        # Log the step before running the query
        self.data_manager.log_step("Before running LLM query (Local)")

        response = llm_runner.run_query(context=context, query=query)
        print("LLM Response (Local):", response)
        
        # Update data object with the response and log the step
        self.data_manager.data_object.llmResult = response
        self.data_manager.log_step("After running LLM query (Local)")

        # Read log file and verify
        log_data = self.read_log_file()
        self.assertIn("Before running LLM query (Local)", log_data[2])
        self.assertIn("After running LLM query (Local)", log_data[-2])

        # Check if the response is correctly added to the data object
        self.assertEqual(self.data_manager.data_object.llmResult, response)

    # def test_run_query_with_openai(self):
    #     api_key = os.getenv("OPENAI_API_KEY", "fake_api_key")
    #     model = "gpt-3.5-turbo"
    #     llm_runner = LLMRunner(api_key=api_key, model=model)

    #     context = self.data_manager.data_object.ragText
    #     query = "What are the key points from the provided text?"

    #     # Log the step before running the query
    #     self.data_manager.log_step("Before running LLM query (OpenAI)")

    #     response = llm_runner.run_query(context=context, query=query)
    #     print("LLM Response (OpenAI):", response)
        
    #     # Update data object with the response and log the step
    #     self.data_manager.data_object.llmResult = response
    #     self.data_manager.log_step("After running LLM query (OpenAI)")

    #     # Read log file and verify
    #     log_data = self.read_log_file()
    #     self.assertIn("Before running LLM query (OpenAI)", log_data[2])
    #     self.assertIn("After running LLM query (OpenAI)", log_data[-2])

    #     # Check if the response is correctly added to the data object
    #     self.assertEqual(self.data_manager.data_object.llmResult, response)

if __name__ == "__main__":
    unittest.main()
