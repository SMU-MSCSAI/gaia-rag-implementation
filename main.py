import json
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gaia_framework.agents.agent_pipeline import Pipeline
import os
from gaia_framework.utils import data_object


app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_file = "./data/data_processing_log.txt"
db_path = "./data/doc_index_db"
    
pipeline = Pipeline(embedding_model_name="text-embedding-ada-002",
                    data_object=data_object,
                    db_type='faiss', 
                    index_type='FlatL2', 
                    chunk_size=100,
                    chunk_overlap=25,
                    log_file=log_file,
                    model_name="llama3")
    
# Allow CORS for your frontend application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    # Process the file here
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

@app.post("/chat/")
async def chat(query: Query):
    # Process the query here
    response = "This is a placeholder response"
    return {"response": response}
  
@app.get("/local/models/")
async def get_supported_models():
    
    try:
        logger.info("Fetching the list of supported models.")
        # Fetch the list of supported models
        valid_models, models = pipeline.load_local_ollama()
        print(models)
        if models:
            return {"models": models}
        else:
            return {"models": []}
    except Exception as e:
        logger.error(f"Error fetching supported models: {e}")
