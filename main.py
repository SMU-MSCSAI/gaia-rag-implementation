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
                    model_name="openchat",
                    file_name="")
pipeline.file_name = f"./data/vdb_{pipeline.embedding_model_name}"
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
        
        if not os.path.exists(file_location):
            # If the file does not exist, upload and save it
            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())
            logger.info(f"File '{file.filename}' saved at '{file_location}'\n\n")
        else:
            logger.info(f"File '{file.filename}' already exists at '{file_location}'. Reading the file...\n\n")

        pipeline.base_path = file_location

        logger.info("Extracting the text data from the source...")
        data_object = pipeline.extract_pdf_data()
        pipeline.data_object = data_object
        if not data_object:
            raise HTTPException(status_code=400, detail="No text data found in the file.")
        
        # Chunk the text and embed once
        logger.info("Chunking the text data...\n\n")
        chunks, data_object = pipeline.process_data_chunk()
        pipeline.data_object = data_object

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks found in the text data.")

        # Embed the chunks and save the vector store
        logger.info("Trying to load the vector store from the local disk...\n\n")
        pipeline.file_name = f"{pipeline.file_name}_{file.filename}"
        if not pipeline.load_vectordb_locall(pipeline.file_name):
            all_embeddings = []
            chunks_metadata = []  # To store metadata of chunks
            for i, chunk in enumerate(chunks):
                embeddings, data_object = pipeline.process_data_embed(chunk.text)
                all_embeddings.append(embeddings)
                pipeline.add_embeddings(embeddings)
                # Save chunk metadata (index and text)
                chunks_metadata.append({"index": i, "text": chunk.text})
            pipeline.data_object = data_object
            pipeline.save_local(pipeline.file_name)

            # Save chunks metadata locally with filename as suffix
            chunks_file = os.path.join(file_base_url, f"chunks_{file.filename}.json")
            with open(chunks_file, 'w') as f:
                print(f"1 chunk: {chunks_metadata[0]}")
                json.dump(chunks_metadata, f)

        response_message = f"File '{file.filename}' processed successfully and stored in the vector store."
        return {"info": response_message}, 201

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")


@app.post("/chat/")
async def chat(query: Query):
    try:
        if len(pipeline.data_object.chunks) < 1:
            # Load chunks from disk if not in memory
            logger.info("Loading chunks from disk...\n\n")
            file_name = pipeline.file_name.split("_")[-1]  # Extract the original file name
            chunks_file = os.path.join(file_base_url, f"chunks_{file_name}.json")
            
            if not os.path.exists(chunks_file):
                raise HTTPException(status_code=404, detail="Chunks file not found.")
            
            with open(chunks_file, 'r') as f:
                chunks_data = json.load(f)
                pipeline.data_object.chunks = [data_object.Chunk(**chunk) for chunk in chunks_data]
          
        logger.info("Embedding the query...\n\n")
        query.query = f"Query: {query.query}, <source>: <{pipeline.data_object.docsSource}>, and <domain>: <{pipeline.data_object.domain}>"
        pipeline.data_object.queries = query.query
        query_embedding = pipeline.process_data_embed(pipeline.data_object.queries)

        logger.info("Searching for the top k most similar embeddings...\n\n")
        similar_results = pipeline.search_embeddings(query_embedding, k=3)
        indices = similar_results.get('indices')[0]

        logger.info("Retrieving the ragText...\n\n")
        ragText = pipeline.retrieveRagText(indices)

        logger.info("Getting the response from the language model...\n\n")
        response = pipeline.run_llm(ragText, pipeline.data_object.queries)
        
        # Format the response to ensure readability
        formatted_response = response.replace('\n', '\n\n')

        return {"response": formatted_response}, 200

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

        pipeline.data_object = data_object  # Ensure data_object is updated
        pipeline.reindex_db(pipeline.db_type, pipeline.index_type)
            
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
