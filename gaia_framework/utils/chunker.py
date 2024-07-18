from dataclasses import dataclass
from gaia_framework.utils.data_object import DataObject
from gaia_framework.utils.logger_util import log_dataobject_step



class TextChunker:
    """
    TextChunker is a class responsible for chunking text into smaller pieces.
    This can be useful for handling large documents and preparing them for processing.
    """
    def __init__(self, chunk_size=512, chunk_overlap=50, separator=" "):
        self.chunk_size = chunk_size  # Maximum size of each chunk
        self.chunk_overlap = chunk_overlap  # Number of characters to overlap between chunks
        self.separator = separator  # Separator to avoid splitting in the middle of a word

    def chunk_text(self, data_object: DataObject, log_file: str = "data_processing_log.txt"):
        """
        Chunk the text in the DataObject and update the DataObject with the list of chunks.
        Log the state of DataObject at each step.
        """
        log_dataobject_step(data_object, "Input Text", log_file)   
        text = data_object.textData
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Ensure that we do not split the chunk in the middle of a word by finding the last occurrence of the separator
            if end < text_length:
                separator_index = chunk.rfind(self.separator)
                if separator_index != -1:
                    end = start + separator_index + 1  # Adjust the end to include the separator
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start += self.chunk_size - self.chunk_overlap

        data_object.chunks = chunks  # Save the chunks as a list in the DataObject

        # Log the state of the DataObject after chunking
        log_dataobject_step(data_object, "After Chunking", log_file)

        return chunks
