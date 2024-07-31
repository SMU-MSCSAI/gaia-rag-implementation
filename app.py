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

global_pipeline = None

class ProjectData(BaseModel):
    id: str
    domain: str
    docsSource: str
    queries: list[str]


@app.post("/initialize_rag/")
async def initialize_rag(project_data: ProjectData):
    global global_pipeline
    try:
        data_object = data_object.DataObject(
            id=project_data.id,
            domain=project_data.domain,
            docsSource=project_data.docsSource,
            queries=project_data.queries,
            textData="",  # This will be filled later when processing files
            chunks=[],
            embedding=None,
            vectorDB=None,
            ragText=None,
            llmResult=None
        )

        global_pipeline = Pipeline(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            data_object=data_object,
            db_type="faiss",
            index_type="FlatL2",
            chunk_size=512,
            chunk_overlap=50,
            base_path="./data",
            log_file="./data/data_processing_log.txt",
            model_name="openchat"
        )

        return {"message": "RAG initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing RAG: {str(e)}")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global global_pipeline
    if not global_pipeline:
        raise HTTPException(status_code=400, detail="Please initialize RAG first")

    try:
        file_location = f"{global_pipeline.base_path}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        
        global_pipeline.data_object = global_pipeline.extract_pdf_data()
        chunks, global_pipeline.data_object = global_pipeline.process_data_chunk()
        
        db_path = f"./data/doc_index_db_{global_pipeline.embedding_model_name}"
        if not global_pipeline.load_vectordb_locall(db_path):
            for chunk in global_pipeline.data_object.chunks:
                embeddings, global_pipeline.data_object = global_pipeline.process_data_embed(chunk.text)
                global_pipeline.add_embeddings(embeddings)
            global_pipeline.save_local(db_path)

        return {"message": f"File '{file.filename}' uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/chat/")
async def chat(query: Query):
    global global_pipeline
    if not global_pipeline:
        raise HTTPException(status_code=400, detail="Please initialize RAG first")

    try:
        query_embedding, _ = global_pipeline.process_data_embed(query.query)
        similar_results = global_pipeline.search_embeddings(query_embedding, k=5)
        indices = similar_results.get('indices')[0]
        rag_text = global_pipeline.retrieveRagText(indices)
        response = global_pipeline.run_llm(rag_text, query.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat query: {str(e)}")

# Streamlit frontend (streamlit_app.py)

import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"  # Adjust as needed

def main():
    st.title("Mini Custom RAG Pipeline")

    # Initialize RAG section
    st.header("1. Initialize RAG")
    
    id_input = st.text_input("Enter ID/Name:")
    domain_input = st.text_input("Enter Domain URL:")
    docs_source = st.selectbox("Select Document Source:", ["web", "pdf", "database", "api"])
    queries_input = st.text_area("Enter Initial Queries (one per line):")

    if st.button("Initialize RAG"):
        queries_list = [query.strip() for query in queries_input.split("\n") if query.strip()]
        
        project_data = {
            "id": id_input,
            "domain": domain_input,
            "docsSource": docs_source,
            "queries": queries_list
        }

        response = requests.post(f"{BACKEND_URL}/initialize_rag/", json=project_data)
        if response.status_code == 200:
            st.success("RAG initialized successfully")
            st.session_state.rag_ready = True
        else:
            st.error(f"Failed to initialize RAG: {response.text}")

    # File Upload section
    st.header("2. Upload File")
    
    if 'rag_ready' in st.session_state and st.session_state.rag_ready:
        uploaded_file = st.file_uploader("Choose a file", type="pdf")
        if uploaded_file is not None:
            files = {"file": uploaded_file}
            response = requests.post(f"{BACKEND_URL}/upload/", files=files)
            if response.status_code == 200:
                st.success("File uploaded and processed successfully")
            else:
                st.error(f"Failed to upload file: {response.text}")

    # Query RAG section
    st.header("3. Query RAG")

    if 'rag_ready' in st.session_state and st.session_state.rag_ready:
        user_query = st.text_input("Enter your query:")
        if st.button("Submit Query"):
            if user_query:
                response = requests.post(f"{BACKEND_URL}/chat/", json={"query": user_query})
                if response.status_code == 200:
                    result = response.json()
                    st.write("RAG Result:")
                    st.write(result['response'])
                else:
                    st.error(f"Failed to query RAG: {response.text}")
            else:
                st.warning("Please enter a query.")
    else:
        st.warning("Please initialize the RAG system first.")

if __name__ == "__main__":
    main()