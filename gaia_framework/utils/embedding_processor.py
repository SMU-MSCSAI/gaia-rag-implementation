import torch
from transformers import AutoTokenizer, AutoModel
import openai
import logging

class EmbeddingProcessor:
    SUPPORTED_MODELS = {
        "huggingface": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "bert-base-uncased",
            "roberta-base",
            # Add more Hugging Face models as needed
        ],
        "openai": [
            "openai/text-embedding-ada-002"
            # Add more OpenAI models as needed
        ]
    }

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        try:
            if model_name in self.SUPPORTED_MODELS["huggingface"]:
                self.logger.info(f"Initializing Hugging Face model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
            elif model_name in self.SUPPORTED_MODELS["openai"]:
                self.logger.info(f"Initializing OpenAI model: {model_name}")
                openai.api_key = "YOUR_OPENAI_API_KEY"  # Set your OpenAI API key
            else:
                self.logger.error(f"Unsupported model: {model_name}")
                raise ValueError(f"Unsupported model. Supported models: {self.SUPPORTED_MODELS}")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def embed_text(self, text):
        self.logger.info(f"Embedding text using model: {self.model_name}")
        try:
            if self.model_name in self.SUPPORTED_MODELS["huggingface"]:
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, padding=True
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.squeeze().numpy()
            elif self.model_name in self.SUPPORTED_MODELS["openai"]:
                response = openai.Embedding.create(
                    model=self.model_name,
                    input=text
                )
                embeddings = response['data'][0]['embedding']
                return embeddings
            else:
                self.logger.error(f"Unsupported model during embedding: {self.model_name}")
                raise ValueError(f"Unsupported model. Supported models: {self.SUPPORTED_MODELS}")
        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            raise
