import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import logging
from dotenv import load_dotenv
import os
from gaia_framework.utils.data_object import DataObject
from gaia_framework.utils.logger_util import log_dataobject_step

class EmbeddingProcessor:
    SUPPORTED_MODELS = {
        "huggingface": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "bert-base-uncased",
            "roberta-base",
            # Add more Hugging Face models as needed
        ],
        "openai": ["text-embedding-ada-002", "text-embedding-babbage-001"],
    }

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingProcessor with the specified model.

        Args:
            model_name (str): The name of the model to use for embeddings.
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Load environment variables from .env file if it exists
        load_dotenv()

        try:
            # Initialize Hugging Face model
            if model_name in self.SUPPORTED_MODELS["huggingface"]:
                self.logger.info(f"Initializing Hugging Face model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)

            # Initialize OpenAI model
            elif model_name in self.SUPPORTED_MODELS["openai"]:
                self.logger.info(f"Initializing OpenAI model: {model_name}")
                # Ensure your OpenAI API key is set in environment variables
                openai_key = os.getenv("OPENAI_API_KEY")
                if openai_key is None:
                    self.logger.error("OpenAI API key not found in environment variables.")
                    raise ValueError("OpenAI API key not found in environment variables.")

                self.client = OpenAI(api_key=openai_key)

            # Handle unsupported model
            else:
                self.logger.error(f"Unsupported model: {model_name}")
                raise ValueError(f"Unsupported model. Supported models: {self.SUPPORTED_MODELS}")

        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def embed_text(self, data_object, text: str, log_file: str = "embedding_log.txt"):
        """
        Generate embeddings for the given text using the specified model.

        Args:
            data_object (DataObject): The data object containing the text to generate embeddings for.
            log_file (str): The file to log processing steps.

        Returns:
            numpy.ndarray or list and dict: The generated embeddings and data object.
        """
        self.logger.info(f"Embedding text using model: {self.model_name}")
        try:
            log_dataobject_step(data_object, "Input Logs to Embeddings Agent:", log_file)
            
            if self.model_name in self.SUPPORTED_MODELS["huggingface"]:
                # Tokenize and generate embeddings using Hugging Face model
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                data_object.embedding = embeddings.tolist()
                log_dataobject_step(data_object, "After Hugging Face Embeddings Agent", log_file)
                return embeddings, data_object

            elif self.model_name in self.SUPPORTED_MODELS["openai"]:
                # Generate embeddings using OpenAI model with the new API
                response = self.client.embeddings.create(
                    model=self.model_name, input=text
                )
                input = [text]  # Note the change here to wrap text in a list)
                embeddings = response.data[0].embedding
                data_object.embedding = embeddings
                log_dataobject_step(data_object, "After OpenAI Embeddings Agent", log_file)
                return embeddings, data_object

            else:
                self.logger.error(f"Unsupported model during embedding: {self.model_name}")
                raise ValueError(f"Unsupported model. Supported models: {self.SUPPORTED_MODELS}")

        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            log_dataobject_step(data_object, f"Error embedding text: {e}", log_file)
            raise
