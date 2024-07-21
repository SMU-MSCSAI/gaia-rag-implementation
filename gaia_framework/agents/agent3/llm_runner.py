import os
from typing import Optional, List
import ollama
from dataclasses import dataclass
from openai import OpenAI
import requests

@dataclass
class LLMRunner:
    api_key: Optional[str] = None
    local_endpoint: Optional[str] = None
    model: Optional[str] = None
    supported_local_models: List[str] = None

    def __post_init__(self):
        if self.api_key:
            # Ensure your OpenAI API key is set in environment variables
            openai_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=openai_key)
        
        if self.local_endpoint:
            self.supported_local_models = self.get_supported_local_models()
            self.ollama_client = ollama.Client()
        self.system_prompt = (
            """You are an advanced language model assistant. Use the context provided to answer the query accurately.
                Make sure you understand the context before generating the response. Explain how you come to your findings.
                Reference the specific sections of the context where the facts are gathered. The context is as follows:
            """
        )


    def get_supported_local_models(self) -> List[str]:
        """
        Get a list of supported local models from the Ollama server.

        Returns:
            List[str]: List of supported local model names.
        """
        try:
            response = requests.get(f"{self.local_endpoint}/v1/models")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching supported models: {e}")
            return []

    def run_query(self, context: str, query: str) -> str:
        """
        Run a query with the provided context using the specified LLM.

        Args:
            context (str): The context text to provide to the LLM.
            query (str): The query to run.

        Returns:
            str: The LLM's response to the query.
        """

        if self.model:
            if self.model in self.supported_local_models:
                try:
                    response = self.ollama_client.generate(
                        model=self.model,
                        prompt=f"{self.system_prompt}\n\nContext: {context}\n\nQuery: {query}"
                    )
                    return response['response']
                except ollama.OllamaError as e:
                    return f"Error with Ollama request: {e}"
            else:
                return f"Model {self.model} is not in the list of supported local models."
        elif self.api_key and self.model:
            response = self.client.completions.create(
                model=self.model,
                prompt=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Context: {context}"},
                    {"role": "user", "content": f"Query: {query}"}
                ]
            )
            return response.choices[0]['message']['content']
        else:
            raise ValueError("No valid API key for OpenAI or local endpoint provided for LLM.")
