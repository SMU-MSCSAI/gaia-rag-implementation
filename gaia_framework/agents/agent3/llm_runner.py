import logging
import os
import time
from typing import Optional, List
import ollama
from dataclasses import dataclass
from openai import OpenAI
import requests
from tqdm import tqdm

from gaia_framework.utils.data_object import DataObject
from gaia_framework.utils.logger_util import log_dataobject_step

@dataclass
class LLMRunner:
    def __init__(
        self,
        api_key: Optional[str] = None,
        local_endpoint: Optional[str] = None,
        model: Optional[str] = None,
        data_object: DataObject = None,
        log_file: str = "data_processing_log.txt",
    ):
        self.api_key = api_key
        self.local_endpoint = local_endpoint
        self.model = model
        self.ollama_client = ollama.Client() if self.model else None
        self.openai_client = (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if api_key and model else None
        )
        self.system_prompt = """You are an advanced language model assistant. Use the context provided to answer the query accurately.
                Make sure you understand the context before generating the response. Explain how you come to your findings.
                Make your response concise and informative. Provide references to the context where the facts are gathered.
            """
        self.data_object = data_object
        self.log_file = log_file
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_supported_local_models(self) -> List[str]:
        """
        Get a list of supported local models from the Ollama server.

        Returns:
            List[str]: List of supported local model names with their parameter sizes.
        """
        try:
            self.logger.info("Fetching supported models from Ollama server.")
            response = self.ollama_client.list()
            models = response.get("models", [])

            model_names = [
                # f"{model.get('name')}_{model.get('details', {}).get('parameter_size')}"
                (model.get("name"), model.get("details", {}).get("parameter_size"))
                for model in models
            ]
            return model_names
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching supported models: {e}")
            print(f"Error fetching supported models: {e}")
            return []

    def download_model(self, model_name: str) -> str:
        """
        Download the specified model from the Ollama server.

        Args:
            model_name (str): The name of the model to download.

        Returns:
            str: The path to the downloaded model.
        """
        try:
            self.logger.info(f"Downloading model: {model_name}")

            dots = ""
            max_dots = 10
            pbar = tqdm(total=100, desc=f"Downloading {model_name}..........", unit='%', ncols=100, bar_format='{desc}')
            while True:
                response = self.ollama_client.pull(model=model_name)
                if response.get('status') == 'success':
                    pbar.n = 100
                    pbar.refresh()
                    pbar.close()
                    break

                # Update dots for the progress bar
                dots = (dots + ".") if len(dots) < max_dots else "."
                desc = f"Downloading {model_name}{dots.ljust(max_dots)}.........."
                pbar.set_description(desc)
                pbar.refresh()

                time.sleep(1)  # Polling interval
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading model: {e}")
            print(f"Error downloading model: {e}")
            return ""

    def run_query(self, context: str, query: str) -> str:
        """
        Run a query with the provided context using the specified LLM.

        Args:
            context (str): The context text to provide to the LLM.
            query (str): The query to run.

        Returns:
            str: The LLM's response to the query.
        """
        self.logger.info(f"Running query with context and query: {query}")
        if self.model:
            log_dataobject_step(
                self.data_object, "Input Text to LLM Agent", self.log_file
            )
            try:
                self.logger.info(f"Generating response using Ollama model: {self.model}")
                response = self.ollama_client.generate(
                    model=self.model,
                    prompt=f"{self.system_prompt}\n\nContext: {context}\n\nQuery: {query}",
                )
                self.data_object.generatedResponse = response["response"]
                log_dataobject_step(
                    self.data_object, "After LLM Response Generated", self.log_file
                )
                return response["response"]
            except ollama.OllamaError as e:
                self.logger.error(f"Error with Ollama request: {e}")
                return f"Error with Ollama request: {e}"
        elif self.api_key and self.model:
            self.logger.info(f"Generating response using OpenAI model: {self.model}")
            response = self.client.completions.create(
                model=self.model,
                prompt=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Context: {context}"},
                    {"role": "user", "content": f"Query: {query}"},
                ],
            )
            return response.choices[0]["message"]["content"]
        else:
            self.logger.error(
                "No valid API key for OpenAI or local endpoint provided for LLM."
            )
            raise ValueError(
                "No valid API key for OpenAI or local endpoint provided for LLM."
            )
