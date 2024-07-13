from typing import Optional, List
from dataclasses import dataclass, field
import json

@dataclass
class DataObject:
    """
    DataObject is a class representing a data structure for holding 
    various information about documents, queries, embeddings, and more.
    """
    id: str  # Unique identifier for the data object
    domain: str  # Domain related to the data object
    docsSource: str  # Source of the documents
    queries: Optional[List[str]] = field(default_factory=list)  # List of queries associated with the data object
    textData: Optional[str] = None  # The raw text data, such as scraped or loaded pdf text
    embedding: Optional[str] = None  # Embedding information if available
    vectorDB: Optional[str] = None  # Vector database information if available
    ragText: Optional[str] = None  # RAG (Retrieval-Augmented Generation) text if available
    chunks: Optional[List[str]] = field(default_factory=list)  # List of text chunks

    def to_dict(self):
        """
        Convert the DataObject to a dictionary.
        Only include non-None values in the dictionary.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a DataObject instance from a dictionary.
        Only include keys that match the class annotations.
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_json(self):
        """
        Convert the DataObject to a JSON string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """
        Create a DataObject instance from a JSON string.
        """
        return cls.from_dict(json.loads(json_str))
