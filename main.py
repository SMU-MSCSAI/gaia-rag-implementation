import json
import logging
from urllib.parse import unquote
from fastapi import FastAPI, File, UploadFile, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gaia_framework.agents.agent_pipeline import Pipeline
import os
from gaia_framework.utils import data_object

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_file = "./data/data_processing_log.txt"

file_base_url = "./files/"
global embedding_model
embedding_model = "text-embedding-ada-002"

data_object = data_object.DataObject(id="1", domain="example.com", docsSource="source", textData="")

pipeline = Pipeline(embedding_model_name=embedding_model,
                    data_object=data_object,
                    db_type='faiss', 
                    index_type='FlatL2', 
                    chunk_size=100,
                    chunk_overlap=25,
                    base_path=file_base_url,
                    log_file=log_file,
                    model_name="llama3")
db_path = f"./data/doc_index_db_{pipeline.embedding_model_name}"
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
    try:
        file_location = f"{pipeline.base_path}/{file.filename}"
        pipeline.base_path = file_location
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
            logger.info(f"File '{file.filename}' saved at '{file_location}'\n\n")
            
            logger.info("Extracting the text data from the source...")
            data_object = pipeline.extract_pdf_data()
            if not data_object:
                raise HTTPException(status_code=400, detail="No text data found in the file.")
              
            # 1. Chunk the text
            logger.info("Chunking the text data...\n\n")
            chunks, data_object = pipeline.process_data_chunk()

            if not chunks:
                raise HTTPException(status_code=400, detail="No chunks found in the text data.")
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
            response_message = f"File '{file.filename}' saved at '{file_location}' and stored in the vector store."
        return {"info": response_message}, 201
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")

@app.post("/chat/")
async def chat(query: Query):
    try:
        response = "This is a placeholder response"
        # embed the query
        query_text = pipeline.add_embeddings(query.query)
        
        # search the index
        
        return {"response": response}, 200
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat query: {e}")

@app.get("/local/models/")
async def get_supported_models():
    try:
        logger.info("Fetching the list of supported models.")
        valid_models, models = pipeline.load_local_ollama()
        if models:
            return {"models": models}, 200
        else:
            return {"models": []}, 204
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error fetching supported models: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching supported models: {e}")

@app.get("/supported_models/")
async def get_supported_embeddings():
    try:
        supported_embeddings = pipeline.embedding_dimensions
        models = list(supported_embeddings.keys())
        model_list = {index: model for index, model in enumerate(models)}
        return {"supported_models": model_list}, 200
    except Exception as e:
        logger.error(f"Error fetching supported models: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching supported models: {e}")

@app.post("/embedding/")
async def set_embedding_model(index: int):
    try:
        supported_embeddings = pipeline.embedding_dimensions
        models = list(supported_embeddings.keys())
        if index < 0 or index >= len(models):
            raise HTTPException(status_code=400, detail=f"Unsupported model index. Valid indices: {list(range(len(models)))}")
        model_name = models[index]
        pipeline.embedding_model_name = model_name
        return {"message": f"Embedding model set to {model_name}"}, 200
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error setting embedding model: {e}")
        raise HTTPException(status_code=500, detail=f"Error setting embedding model: {e}")

@app.get("/current_config/")
async def get_current_config():
    try:
        config = {
            "embedding_model_name": pipeline.embedding_model_name,
            "db_type": pipeline.db_type,
            "index_type": pipeline.index_type,
            "chunk_size": pipeline.chunk_size,
            "chunk_overlap": pipeline.chunk_overlap,
            "base_path": pipeline.base_path,
            "log_file": pipeline.log_file,
            "model_name": pipeline.model_name,
        }
        return config, 200
    except Exception as e:
        logger.error(f"Error fetching current configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching current configuration: {e}")

@app.post("/config/")
async def set_config(config: dict):
    try:
        if "embedding_model_name" in config:
            pipeline.embedding_model_name = config["embedding_model_name"]
        if "db_type" in config:
            pipeline.db_type = config["db_type"]
        if "index_type" in config:
            pipeline.index_type = config["index_type"]
        if "chunk_size" in config:
            pipeline.chunk_size = config["chunk_size"]
        if "chunk_overlap" in config:
            pipeline.chunk_overlap = config["chunk_overlap"]
        if "base_path" in config:
            pipeline.base_path = config["base_path"]
        if "log_file" in config:
            pipeline.log_file = config["log_file"]
        if "model_name" in config:
            pipeline.model_name = config["model_name"]
        
        pipeline.log_file = config["log_file"]
        pipeline.data_object = data_object
            
        config = {
            "embedding_model_name": pipeline.embedding_model_name,
            "db_type": pipeline.db_type,
            "index_type": pipeline.index_type,
            "chunk_size": pipeline.chunk_size,
            "chunk_overlap": pipeline.chunk_overlap,
            "base_path": pipeline.base_path,
            "log_file": pipeline.log_file,
            "model_name": pipeline.model_name,
        }
        
        response_message = f"Configuration updated successfully new configuration is: {config}"
        return {"message": response_message}, 200
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {e}")
