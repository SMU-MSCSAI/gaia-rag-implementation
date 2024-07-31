import os
import streamlit as st
import requests
import glob

BACKEND_URL = "http://localhost:8000"  # Adjust as needed

DEFAULT_CONFIG = {
    "id": "default_id",
    "domain": "default.com",
    "docsSource": "pdf",
    "queries": ["What is the default query?"],
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "db_type": "faiss",
    "index_type": "FlatL2",
    "chunk_size": 100,
    "chunk_overlap": 25,
    "model_name": "openchat",
    "top_k": 3
}

def get_supported_models():
    response = requests.get(f"{BACKEND_URL}/supported_models/")
    if response.status_code == 200:
        models_data = response.json()[0]["supported_models"]
        models_list = [model for key, model in models_data.items()]
        return models_list
    else:
        st.error("Failed to fetch supported models.")
        return []

def get_current_config():
    response = requests.get(f"{BACKEND_URL}/current_config/")
    if response.status_code == 200:
        config_data = response.json()[0]
        return config_data
    else:
        st.error("Failed to fetch current configuration.")
        return {}

def set_config(config):
    response = requests.post(f"{BACKEND_URL}/config/", json=config)
    return response

def initialize_defaults():
    response = set_config(DEFAULT_CONFIG)
    if response.status_code == 200:
        st.session_state.rag_ready = True
        st.session_state.config = DEFAULT_CONFIG
        st.success("Default RAG initialized successfully")

def chunks_exist():
    chunk_files = glob.glob("files/chunks_*.json")
    return len(chunk_files) > 0

def main():
    st.title("Mini Custom RAG Pipeline")

    if 'rag_ready' not in st.session_state:
        st.session_state.rag_ready = False
        initialize_defaults()

    with st.expander("Initialize RAG" if not st.session_state.rag_ready else "Reinitialize RAG", expanded=True):
        id_input = st.text_input("Enter ID/Name:", value=st.session_state.config.get("id", "example_id"))
        domain_input = st.text_input("Enter Domain URL:", value=st.session_state.config.get("domain", "example.com"))

        if "http" in domain_input:
            docs_source = "web"
            st.selectbox("Select Document Source:", ["web"], disabled=True, index=0, format_func=lambda x: f"{x} (auto-selected based on URL)")
        else:
            docs_source = st.selectbox("Select Document Source:", ["pdf"], index=0, format_func=lambda x: f"{x} (default)" if x == "pdf" else f"{x} (not currently supported)", disabled=True)
            
        queries_input = st.text_area("Enter Initial Queries (one per line):", value="\n".join(st.session_state.config.get("queries", [])))

        st.subheader("Configure Pipeline")
        
        supported_embeddings = get_supported_models()
        embedding_model_name = st.selectbox("Embedding Model Name", supported_embeddings, index=supported_embeddings.index(st.session_state.config.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")))
        db_type = st.selectbox("Database Type", ["faiss"], index=["faiss"].index(st.session_state.config.get("db_type", "faiss")), disabled=True, format_func=lambda x: f"{x} (only choice)")
        index_type = st.selectbox("Index Type", ["FlatL2", "IVFFlat"], index=["FlatL2", "IVFFlat"].index(st.session_state.config.get("index_type", "FlatL2")))
        chunk_size = st.number_input("Chunk Size", value=st.session_state.config.get("chunk_size", 100))
        chunk_overlap = st.number_input("Chunk Overlap", value=st.session_state.config.get("chunk_overlap", 25))
        model_name = st.selectbox("Model Name", ["openchat", "other_model"], index=["openchat", "other_model"].index(st.session_state.config.get("model_name", "openchat")), format_func=lambda x: f"{x}" if x != "other_model" else f"{x} (not currently supported)")
        top_k = st.number_input("Top K", value=st.session_state.config.get("top_k", 3))

        if st.button("Initialize RAG" if not st.session_state.rag_ready else "Reinitialize RAG"):
            queries_list = [query.strip() for query in queries_input.split("\n") if query.strip()]
            
            project_data = {
                "id": id_input,
                "domain": domain_input,
                "docsSource": docs_source,
                "queries": queries_list,
                "embedding_model_name": embedding_model_name,
                "db_type": db_type,
                "index_type": index_type,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "model_name": model_name,
                "top_k": top_k
            }

            response = set_config(project_data)
            if response.status_code == 200:
                st.success("RAG initialized successfully" if not st.session_state.rag_ready else "RAG reinitialized successfully")
                st.session_state.rag_ready = True
                st.session_state.config = project_data  # Update the session state with new config
            else:
                st.error(f"Failed to initialize RAG: {response.text}")

    # File Upload section
    st.header("2. Upload File")
    
    file_uploaded = chunks_exist()
    if st.session_state.rag_ready:
        if "http" in domain_input:
            st.warning("Since the domain contains 'http', the system will scrape the web for data.")
            file_uploaded = True
        else:
            uploaded_file = st.file_uploader("Choose a file", type="pdf")
            if uploaded_file is not None:
                file_exists = chunks_exist()
                if file_exists:
                    file_uploaded = True
                    st.warning("This file already exists. Re-uploading will delete the existing file and re-process it.")
                if st.button("Upload File"):
                    with st.spinner('Uploading and processing file...'):
                        files = {"file": uploaded_file}
                        response = requests.post(f"{BACKEND_URL}/upload/", files=files)
                        if response.status_code == 200:
                            st.success("File uploaded and processed successfully")
                            file_uploaded = True
                        else:
                            st.error(f"Failed to upload file: {response.text}")

    # Query RAG section
    st.header("3. Query RAG")

    if file_uploaded:
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
        st.warning("Please upload a file first, or provide a domain URL to initialize RAG to be scraped")

if __name__ == "__main__":
    main()
