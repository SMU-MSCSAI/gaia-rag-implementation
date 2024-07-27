import json
import logging
from typing import Optional
from urllib.parse import unquote
import uuid
from fastapi import FastAPI, File, Request, UploadFile, HTTPException, Path
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
                    file_name="",
                    top_k=3)
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
    history_limit: Optional[int] = 5
    
class Config(BaseModel):
    embedding_model_name: Optional[str] = "text-embedding-ada-002"
    db_type: Optional[str] = "faiss"
    index_type: Optional[str] = "FlatL2"
    chunk_size: Optional[int] = 150
    chunk_overlap: Optional[int] = 50
    base_path: Optional[str] = "./files/"
    log_file: Optional[str] = "./data/data_processing_log.txt"
    model_name: Optional[str] = "llama3"
    top_k: Optional[int] = 5

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Construct the correct file location
        file_location = os.path.join(pipeline.base_path, file.filename)
        chunks_file = os.path.join(file_base_url, f"chunks_{file.filename}.json")
        pipeline.file_name = file.filename
        
        # Check if the file and its chunks already exist
        if os.path.exists(file_location) and os.path.exists(chunks_file):
            logger.info(f"File '{file.filename}' and its chunk file already exist at '{file_location}'. Skipping processing...\n\n")
        else:
            # If the file does not exist, upload and save it
            if not os.path.exists(file_location):
                with open(file_location, "wb+") as file_object:
                    file_object.write(file.file.read())
                logger.info(f"File '{file.filename}' saved at '{file_location}'\n\n")
            else:
                logger.info(f"File '{file.filename}' already exists at '{file_location}'. Reading the file...\n\n")

            # Set the base path to the file location
            pipeline.base_path = f"{file_base_url}/{file.filename}"

            # Initialize data object
            pipeline.data_object.id = str(uuid.uuid4())

            # Process the PDF and extract text data
            logger.info("Extracting the text data from the source...")
            data_object = pipeline.extract_pdf_data()
            if not data_object or not data_object.textData:
                raise HTTPException(status_code=400, detail="No text data found in the file.")
            
            # Chunk the text and embed once
            logger.info("Chunking the text data...\n\n")
            chunks, data_object = pipeline.process_data_chunk()
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
                pipeline.save_local(f"./data/{pipeline.file_name}")

                # Save chunks metadata locally with filename as suffix
                with open(chunks_file, 'w') as f:
                    json.dump(chunks_metadata, f)

        response_message = f"File '{file.filename}' processed successfully and stored in the vector store."
        return {"info": response_message}, 201

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")


@app.post("/chat/")
async def chat(request: Request):
    try:
        req = await request.json()
        if not req:
            raise HTTPException(status_code=400, detail="No query provided.")
        
        query: Query = Query(**req)
        logger.info(f"Received query: {query.query}\n\n")
        
        if len(pipeline.data_object.chunks) < 1:
            # Load chunks from disk if not in memory
            logger.info("Loading chunks from disk...\n\n")
            file_name = pipeline.file_name.split("_")[-1]  # Extract the original file name
            chunks_file = os.path.join(file_base_url, f"chunks_{file_name}.json")
            
            print(f"chunk file: {chunks_file}")
            if not os.path.exists(chunks_file):
                raise HTTPException(status_code=404, detail="Chunks file not found.")
            
            with open(chunks_file, 'r') as f:
                chunks_data = json.load(f)
                pipeline.data_object.chunks = [value.get("text") for value in chunks_data]
            
        logger.info("Embedding the query...\n\n")
        embedded_query = f"Query: {query.query}, <source>: <{pipeline.data_object.docsSource}>, and <domain>: <{pipeline.data_object.domain}>"
        pipeline.data_object.queries = embedded_query
        query_embedding = pipeline.process_data_embed(pipeline.data_object.queries)

        logger.info(f"Searching for the top {pipeline.top_k} most similar embeddings...\n\n")
        similar_results = pipeline.search_embeddings(query_embedding)
        indices = similar_results.get('indices')[0]

        logger.info("Retrieving the ragText...\n\n")
        ragText = pipeline.retrieveRagText(indices)

        logger.info("Getting the response from the language model...\n\n")
        try:
            response = pipeline.run_llm(ragText, query.query, history_limit=query.history_limit)  # Use original query here
            pipeline.data_object.conversation_history.append(pipeline.conversation_history[-query.history_limit:])
        except Exception as llm_error:
            logger.error(f"LLM error: {str(llm_error)}")
            raise HTTPException(status_code=500, detail=f"Error in language model processing: {str(llm_error)}")
        
        # Format the response to ensure readability
        if isinstance(response, dict):
            formatted_response = json.dumps(response, indent=2).replace('\n', '\n\n')
        else:
            formatted_response = response.replace('\n', '\n\n')

        conversational_history_len = len(pipeline.conversation_history)
        
        return {
            "response": formatted_response,
            "conversation_history_length": conversational_history_len
        }, 200
        
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
        config: Config = {
            "embedding_model_name": pipeline.embedding_model_name,
            "db_type": pipeline.db_type,
            "index_type": pipeline.index_type,
            "chunk_size": pipeline.chunk_size,
            "chunk_overlap": pipeline.chunk_overlap,
            "base_path": pipeline.base_path,
            "log_file": pipeline.log_file,
            "model_name": pipeline.model_name,
            "top_k": pipeline.top_k
        }
        return config, 200
    except Exception as e:
        logger.error(f"Error fetching current configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching current configuration: {e}")

@app.post("/config/")
async def set_config(request: Request):
    try:
        # Update the pipeline configuration
        config = await request.json()
        
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
        if "top_k" in config:
            pipeline.top_k = config["top_k"]

        pipeline.data_object = data_object  # Ensure data_object is updated
        config: Config = {
            "embedding_model_name": pipeline.embedding_model_name,
            "db_type": pipeline.db_type,
            "index_type": pipeline.index_type,
            "chunk_size": pipeline.chunk_size,
            "chunk_overlap": pipeline.chunk_overlap,
            "base_path": pipeline.base_path,
            "log_file": pipeline.log_file,
            "model_name": pipeline.model_name,
            "top_k": pipeline.top_k
        }
        
        pipeline.reindex_db(pipeline.db_type, pipeline.index_type)
        
        response_message = f"Configuration updated successfully new configuration is: {config}"
        return {"message": response_message}, 200
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {e}")

@app.post("/clear_history/")
async def clear_history():
    pipeline.clear_conversation_history()
    return {"message": "Conversation history cleared"}, 200