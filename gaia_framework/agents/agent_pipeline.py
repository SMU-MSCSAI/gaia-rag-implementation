# The `Pipeline` class in the provided Python code handles text processing tasks such as chunking,
# embedding, vector database operations, language model execution, and PDF text extraction in a
# structured pipeline fashion.
import logging
import os
import numpy as np
from gaia_framework.agents.agent3.llm_runner import LLMRunner
from gaia_framework.agents.agent2.vector_db_agent import VectorDatabase
from gaia_framework.utils.chunker import TextChunker
from gaia_framework.utils.data_object import DataObject
from gaia_framework.utils.embedding_processor import EmbeddingProcessor
from gaia_framework.utils.logger_util import log_dataobject_step, reset_log_file
from gaia_framework.agents.agent1.data_collector import DataCollector
from gaia_framework.utils.promptchain import CustomPromptChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        embedding_model_name,
        data_object: DataObject,
        db_type="faiss",
        index_type="FlatL2",
        chunk_size=512,
        chunk_overlap=50,
        log_file="data_processing_log.txt",
        model_name="llamma2",
        base_path: str = "./data",
        local_endpoint=None,
        api_key=None,
        file_name="",
        top_k=3,
    ):
        """
            Usage of the Pipeline class.
            1. Get the data as text to be used as the context.
            2. Embed the text data in chunks.
            3. Initialize the vector database and add the embeddings.
            4. Take a query and embed it.
            5. Search for the top k most similar embeddings from the vector database.
            6. Return the results to be used as the context for the language model.
            7. Run the language model with the context and query.
            8. Get the response from the language model.

        Args:
            dimension (_type_): dimension of the embeddings.
            db_type (str, optional): database type. Defaults to 'faiss'.
            index_type (str, optional): index type. Defaults to 'FlatL2'.
            log_file (str, optional): log file. Defaults to "data_processing_log.txt".
        """
        # Embedding dimensions for each model
        self.embedding_dimensions = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "bert-base-uncased": 768,
            "roberta-base": 768,
            "text-embedding-ada-002": 1536,
            "text-embedding-babbage-001": 2048,
            "text-embedding-3-small": 1536,
            # Add more models and their dimensions as needed
        }

        # Retrieve the correct dimension for the selected model
        self.dimension = self.embedding_dimensions.get(embedding_model_name)
        if self.dimension is None:
            raise ValueError(
                f"Embedding dimension for model '{embedding_model_name}' not found."
            )

        self.db_type = db_type
        self.index_type = index_type
        self.log_file = log_file
        self.embedding_model_name = embedding_model_name
        self.data_object = data_object
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_local_models = ["llamma2", "openchat", "gpt3", "llama3"]
        self.openai_key = api_key
        self.local_endpoint = local_endpoint
        self.model_name = model_name
        self.base_path = base_path
        self.top_k = top_k
        self.conversation_history = []

        self.data_collector = DataCollector(
            base_path=self.base_path, log_file=self.log_file
        )
        self.chunker = TextChunker(self.chunk_size, self.chunk_overlap, separator=",")
        self.embedder = EmbeddingProcessor(self.embedding_model_name)
        # instantiate the class, and create the index (data structure sufficient to store the embeddings)
        self.vector_db = VectorDatabase(self.dimension, self.db_type, self.index_type)
        # load the llm
        self.llm = LLMRunner(
            api_key=self.openai_key,
            local_endpoint=self.local_endpoint,
            model=self.model_name,
            data_object=self.data_object,
            log_file=self.log_file,
        )

        self.prompt_chain = CustomPromptChain()

        # restart the log file after each run
        reset_log_file(self.log_file)

        # clean up the logging file after each run
        def tearDown(self):
            # Clean up log file after tests
            if os.path.exists(self.log_file):
                os.remove(self.log_file)

    def process_data_chunk(self):
        """
        Process the data by chunking and embedding the text data.

        Args:
            data (list): The list of text data to process.
        """
        # get the chunks and the data object
        chunks, data_object = self.chunker.chunk_text(self.data_object, self.log_file)
        return chunks, data_object

    def process_data_embed(self, text):
        # embed the chunks
        embeddings, data_object = self.embedder.embed_text(
            self.data_object,
            text,
            model_name=self.embedding_model_name,
            log_file=self.log_file,
        )
        self.data_object = data_object
        return embeddings, data_object

    def add_embeddings(self, embeddings):
        """
        Add the embeddings to the vector database.

        Args:
            embeddings (list): The list of embeddings to add.
        """
        self.vector_db.add_embeddings(self.data_object, embeddings, self.log_file)

    def save_local(self, path):
        """
        Persist the vector store to the local disk.

        Args:
            path (str): The path to save the database.
        """
        if not os.path.exists(path):
            self.vector_db.save_local(path, self.data_object, self.log_file)
        else:
            logger.warn(
                f"Path {path} already exists. Please provide a db name to create another index db locally."
            )

    def load_vectordb_locall(self, path):
        """
        Load the vector store from the local disk.

        Args:
            path (str): The path to load the database from.
        """
        if os.path.exists(path):
            self.vector_db.load_local(path, self.data_object, self.log_file)
            return True
        else:
            logger.warning(
                f"Path {path} does not exist. Please provide a valid path to load the index db or save the db first."
            )
            return False

    def search_embeddings(self, query_embedding):
        """
        Search the vector database for the top k most similar embeddings.

        Args:
            query_embedding (np.ndarray): The query embedding to search for.
            k (int): The number of top similar embeddings to return. Default is 5.

        Returns:
            dict: A dictionary containing the distances and indices of the top k similar embeddings.
        """
        similar_results = self.vector_db.get_similarity_indices(
            query_embedding, self.data_object, self.top_k, self.log_file
        )
        return similar_results

    def retrieveRagText(self, indices):
        """
        Get the ragText from the data object using the indices from the similar results.

        Args:
            indices (list): The list of indices to retrieve the ragText from.

        Returns:
            str: The concatenated ragText retrieved from the data object.
        """
        log_dataobject_step(
            self.data_object, "Input Text to RagText Retrieval", self.log_file
        )
        logger.info("Retrieving RagText.")
        try:
            ragText = " ".join([self.data_object.chunks[idx] for idx in indices])
            self.data_object.ragText = ragText
            log_dataobject_step(
                self.data_object, "After RagText Retrieved", self.log_file
            )
            logger.info(f"RagText Retrieved successfully!")
            return ragText
        except IndexError as e:
            logger.error(f"IndexError while retrieving RagText: {e}")
            raise
        except Exception as e:
            logger.error(f"Error while retrieving RagText: {e}")
            raise

    def load_local_ollama(self):
        """
        Load the supported local models.
        """
        local_models = self.llm.get_supported_local_models()
        model_names = [
            model[0].split(":")[0] for model in local_models
        ]  # Adjusted splitting to match your format
        # Check if the local models are among the supported local models
        valid_models = []
        for name in model_names:
            if name in self.supported_local_models:
                valid_models.append(name)
            if (
                self.model_name in self.supported_local_models
                and self.model_name not in model_names
            ):
                # download the ollama model locally
                logger.info(
                    f"Model {self.model_name} not found locally. Downloading it from ollama model registry....\n"
                )
                response = self.llm.download_model(self.model_name)
                model_names.append(self.model_name)
        if valid_models:
            logger.info(f"Valid local models loaded successfully: {valid_models}")
        else:
            logger.warning("No valid local models found.")
        return valid_models, model_names

    def run_llm(self, context, query, history_limit=5):
        logger.info(
            f"Checking if the model exists in the list of supported local models."
        )
        _, model_names = self.load_local_ollama()
        if self.model_name not in model_names:
            logger.error(
                f"Model {self.model_name} not found in the list of supported local models.\n"
            )
            raise ValueError(
                f"Model {self.model_name} not found in the list of supported local models."
            )

        self.llm.model = self.model_name

        # Use CustomPromptChain
        prompt_context = {
            "context": context,
            "query": query,
            # Add any other context variables you need
        }

        prompt_template = """
        User:
            Context: {{ context }} and Conversation History: {{ conversation_history }}
            Query: {{ query }}
        
        Please provide a detailed and accurate response based on the given context (which involves conversational history) and query. Make sure you don't add the conversational history in the response, only use it as a context to generate the response.
        """

        try:
            response = self.prompt_chain.run(
                context=prompt_context,
                model=self.llm.model,
                callable=self.llm.run_query,
                prompt=prompt_template,
                conversation_history=self.conversation_history,
            )

            # Update conversation history
            self.conversation_history.append({"query": query, "response": response})

            # Limit conversation history to last 5 exchanges
            self.conversation_history = self.conversation_history[-history_limit:]

            return response
        except Exception as e:
            logger.error(f"Error running LLM: {str(e)}")
            raise

    def extract_pdf_data(self):
        """
        Extract the text data from a PDF file.

        Args:
            file_path (str): The path to the PDF file to extract text from.
        """
        data_object = self.data_collector.process_pdf(self.base_path, self.data_object)
        return data_object

    def reindex_db(self, db_type, index_type):
        """
        Summary:
            Reindex the database with new embedding model dimensions.
        Args:
            db_type (str): The database type.
            index_type (str): The index type.
        """
        self.dimension = self.embedding_dimensions.get(self.embedding_model_name)
        self.db_type = db_type
        self.index_type = index_type
        logger.info(
            f"Reindexing the database, db_type: {self.db_type}, index_type: {self.index_type} with new embedding model dims: {self.dimension}"
        )
        self.vector_db = VectorDatabase(self.dimension, self.db_type, self.index_type)

    def clear_conversation_history(self):
        self.conversation_history = []


if __name__ == "__main__":
    data = "This is a test sentence about domestic animals, Here I come with another test sentence about the cats."
    data_object = DataObject(
        id="test_id",
        domain="test_domain",
        textData=data,
        docsSource="test_docsSource",
        queries=["what's this test about?"],
        chunks=[],
        embedding=None,
        vectorDB=None,
        ragText=None,
        llmResult=None,
    )

    # embedding_model_name = "text-embedding-ada-002"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    log_file = "./data/data_processing_log.txt"
    db_path = f"./data/doc_index_db_{embedding_model_name}"

    pipeline = Pipeline(
        embedding_model_name=embedding_model_name,
        data_object=data_object,
        db_type="faiss",
        index_type="FlatL2",
        chunk_size=512,
        chunk_overlap=50,
        base_path="./data",
        log_file=log_file,
        model_name="openchat",
    )

    # 0. Extract the text data from a PDF file
    logger.info("Extracting the text data from the source...")
    pdf_file_path = "./files/Google_Cert_Learning Plan.pdf"
    data_object = pipeline.extract_pdf_data(pdf_file_path, data_object)

    # 1. Chunk the text
    logger.info("Chunking the text data...\n\n")
    chunks, data_object = pipeline.process_data_chunk()

    # # 2. If the db is already saved, load it, otherwise add the embeddings and save it
    logger.info("Trying to load the vector store from the local disk...\n\n")
    if not pipeline.load_vectordb_locall(db_path):
        # 2.1 Embed the chunks
        all_embeddings = []
        for chunk in data_object.chunks:
            embeddings, data_object = pipeline.process_data_embed(chunk.text)
            # 2.2 Add embeddings to the vector database
            pipeline.add_embeddings(embeddings)
        # 2.3 Persist the vector store to the local disk or load if exist
        pipeline.save_local(db_path)

    # # 3. Embed the query
    logger.info("Embedding the query...\n\n")
    data_object.queries = [
        "what are the included links? list all of them in a bulletpoint format."
    ]
    query_embedding = pipeline.process_data_embed(data_object.queries[0])

    # # 4. Search for the top k most similar embeddings
    logger.info("Searching for the top k most similar embeddings...\n\n")
    similar_results = pipeline.search_embeddings(query_embedding)
    indices = similar_results.get("indices")[0]  # Extract the first list of indices

    # # 5. Get the ragText from the data object using the indices from the similar results
    # # Iterate over the indices to get the text out of the indexed chunks
    logger.info("Retrieving the ragText...\n\n")
    ragText = pipeline.retrieveRagText(indices)

    # # 6. Run the language model with the context and query
    # logger.info("Running the language model...\n\n")
    # local_models = pipeline.load_local_ollama()

    # # 7. Get the response from the language model
    logger.info("Getting the response from the language model...\n\n")
    response = pipeline.run_llm(data_object.ragText, data_object.queries[0])

    # logger.info(f"Response from the language model: {response}")
    print(f"Response from the language model: {response}")
    logger.info("Pipeline completed successfully, check the log file for more details.")
