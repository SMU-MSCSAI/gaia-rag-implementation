import streamlit as st
import os
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Union
from promptchain import CustomPromptChain
from Agent_1.models import ProjectData

## TO RUN streamlit run app.py     


def build_models():
    load_dotenv()
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    return anthropic_client, openai_client

#I realized that anthropic doesn't like non businesses to run their api
def prompt_anthropic(model: anthropic.Anthropic, prompt: str):
    message = model.messages.create(
        model="claude-3.5",
        max_tokens=1000,
        temperature=0.5,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content

def prompt_openai(model: OpenAI, prompt: str):
    response = model.chat.completions.create(
        model="gpt-4o-Turbo", 
        max_tokens=1000,
        temperature=0.5,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content



def query_rag(query):
    #what to do what to do
    return f"This wont be fun: '{query}'"

def run_prompt_chain(project_data: ProjectData, query: str, rag_result: str):
    model, prompt_func = st.session_state.selected_model_and_func
    
    prompt = """Given the following conversation history, user query, and RAG result, provide a comprehensive response.
    
    Conversation history:
    {{conversation_history}}
    
    User query: '{{query}}'
    RAG result: '{{rag_result}}'
    
    Provide a detailed response that addresses the user's query, incorporates insights from the RAG result, and considers the conversation history. If the user is asking about a previous response, make sure to reference and explain relevant parts of the conversation history.
    
    Format your response in markdown."""
    
    context = project_data.copy()
    context.update({
        'query': query,
        'rag_result': rag_result,
    })
    
    results, filled_prompts = CustomPromptChain.run(context, model, prompt_func, prompt)
    return results, filled_prompts

def main():
    st.title("Mini Custom RAG Pipeline")

    # Initialize session state
    if 'project_data' not in st.session_state:
        st.session_state.project_data = None
    if 'selected_model_and_func' not in st.session_state:
        st.session_state.selected_model_and_func = None

    # Input Section
    st.header("1. Initialize RAG")
    
    id_input = st.text_input("Enter ID/Name:")
    domain_input = st.text_input("Enter Domain URL:")
    
    docs_source_options = ["web", "pdf", "database", "api"]
    docs_source = st.selectbox("Select Document Source:", docs_source_options)
    
    queries_input = st.text_area("Enter Initial Queries (one per line):")

    # Add model selection dropdown
    selected_model = st.selectbox(
        "Select LLM for RAG:",
        ["Anthropic (Claude)", "OpenAI (GPT-4)"]
    )

    if st.button("Initialize RAG"):
        queries_list = [query.strip() for query in queries_input.split("\n") if query.strip()]
        
        project_data = ProjectData(
            id=id_input,
            domain=domain_input,
            docsSource=docs_source,
            queries=queries_list
        )

        st.json(project_data.to_dict())

        # Set up the selected model and function
        anthropic_client, openai_client = build_models()
        if selected_model == "Anthropic (Claude)":
            st.session_state.selected_model_and_func = (anthropic_client, prompt_anthropic)
        else:  # OpenAI
            st.session_state.selected_model_and_func = (openai_client, prompt_openai)

        # call agent 1
        #call: agent(project_data)

        # if rag_initialized:
        #     st.session_state.project_data = project_data
        #     st.session_state.rag_ready = True
        #     st.success(f"RAG initialized with {selected_model}")

    # Query Section
    st.header("2. Query RAG")

    if 'rag_ready' in st.session_state and st.session_state.rag_ready:
        user_query = st.text_input("Enter your query:")
        if st.button("Submit Query"):
            if user_query:
                rag_result = query_rag(user_query)
                st.write("Initial RAG Result:")
                st.write(rag_result)

                chain_results, filled_prompts = run_prompt_chain(st.session_state.project_data, user_query, rag_result)

                st.write("Prompt Chain Results:")
                for i, (result, prompt) in enumerate(zip(chain_results, filled_prompts), 1):
                    st.write(f"Step {i}:")
                    st.write(f"Prompt: {prompt}")
                    st.write(f"Result: {result}")

                # Option to download results
                result_text = CustomPromptChain.to_delim_text_file("prompt_chain_results", chain_results)
                st.download_button(
                    label="Download Prompt Chain Results",
                    data=result_text,
                    file_name="prompt_chain_results.txt",
                    mime="text/plain"
                )

                # Option to download updated project data
                st.download_button(
                    label="Download Updated Project Data",
                    data=st.session_state.project_data.to_json(),
                    file_name="updated_project_data.json",
                    mime="application/json"
                )
            else:
                st.warning("Please enter a query.")
    else:
        st.warning("Please initialize the RAG system first.")

if __name__ == "__main__":
    main()

