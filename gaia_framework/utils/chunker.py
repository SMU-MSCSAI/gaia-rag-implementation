from dataclasses import dataclass, field
import logging
import re

from numpy import copy
from gaia_framework.utils.data_object import DataObject
from gaia_framework.utils.logger_util import log_dataobject_step

@dataclass
class Chunk:
    text: str
    start_index: int
    end_index: int

class TextChunker:
    """
    TextChunker is a class responsible for chunking text into smaller pieces.
    This can be useful for handling large documents and preparing them for processing.
    """
    def __init__(self, chunk_size=512, chunk_overlap=50, separator=" "):
        self.chunk_size = chunk_size  # Maximum size of each chunk
        self.chunk_overlap = chunk_overlap  # Number of characters to overlap between chunks
        self.separator = separator  # Separator to avoid splitting in the middle of a word
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def chunk_text(self, data_object: DataObject, log_file: str = "data_processing_log.txt"):
        """
        Chunk the text in the DataObject and update the DataObject with the list of chunks.
        
        Args:
            data_object (DataObject): The DataObject containing the text to be chunked.
            log_file (str, optional): The file path of the log file. Defaults to "data_processing_log.txt".
        
        Returns:
            tuple: A tuple containing the list of chunks and the updated DataObject.
        
        Raises:
            Exception: If an error occurs during chunking.
        """
        try:
            # Create a new DataObject for logging, copying only necessary fields
            log_dataobject_step(data_object, "Input Text To the Chunking Agent:", log_file)   
            text = data_object.textData
            if not text:
                return []

            chunks = []
            start = 0
            text_length = len(text)
            self.logger.info(f"Chunking text of length {text_length} with chunk size {self.chunk_size} and overlap {self.chunk_overlap}")
            while start < text_length:
                end = start + self.chunk_size
                chunk = text[start:end]

                # Ensure that we do not split the chunk in the middle of a word by finding the last occurrence of the separator
                if end < text_length:
                    separator_index = chunk.rfind(self.separator)
                    if separator_index != -1:
                        end = start + separator_index + 1  # Adjust the end to include the separator
                        chunk = text[start:end]

                chunks.append(Chunk(text=chunk.strip(), start_index=start, end_index=end))
                start += self.chunk_size - self.chunk_overlap
            self.logger.info(f"Finished Chunking: {start}/{text_length}")
            data_object.chunks = [chunk.text for chunk in chunks]  # Save the chunks as a list in the DataObject
            log_dataobject_step(data_object, "After Chunking Agent", log_file)
            return chunks, data_object
        except Exception as e:
            # Handle any exceptions that occur during chunking
            print(f"An error occurred during chunking: {str(e)}")
            return [], data_object
