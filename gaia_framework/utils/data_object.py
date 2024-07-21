import json
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class DataObject:
    id: str
    domain: str
    docsSource: str
    queries: Optional[List[str]] = field(default_factory=list)
    textData: Optional[str] = None
    embedding: Optional[str] = None
    vectorDB: Optional[str] = None
    ragText: Optional[str] = None
    chunks: Optional[List[str]] = field(default_factory=list)
    llmResult: Optional[str] = None
    vectorDBPersisted: Optional[bool] = False
    embeddingAdded: Optional[bool] = False
    vectorDBLoaded: Optional[bool] = False
    similarityIndices: Optional[dict] = None
    generatedResponse: Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        return cls.from_dict(json.loads(json_str))