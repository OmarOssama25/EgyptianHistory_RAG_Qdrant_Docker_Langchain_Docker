import os
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import Qdrant
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient, models
import time
import math
import requests

# App title
st.title("📚 RAG with Ollama, Qdrant, and LangChain")

# Initialize Qdrant client with better error handling
@st.cache_resource
def get_qdrant_client():
    # Check for Qdrant Cloud configuration
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    
    # If Qdrant Cloud credentials are available, use them
    if qdrant_url and qdrant_api_key:
        try:
            client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
            # Test connection
            client.get_collections()
            st.success("Connected to Qdrant Cloud")
            return client
        except Exception as e:
            st.error(f"Failed to connect to Qdrant Cloud: {str(e)}")
    
    # Try local Qdrant (for development)
    try:
        # Try localhost first (for local development without Docker)
        client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=5.0  # Short timeout to fail fast
        )
        # Test connection
        client.get_collections()
        st.success("Connected to local Qdrant")
        return client
    except Exception as local_error:
        try:
            # Try Docker container name (for Docker Compose setup)
            client = QdrantClient(
                host="qdrant",
                port=6333,
                timeout=5.0
            )
            # Test connection
            client.get_collections()
            st.success("Connected to Qdrant in Docker")
            return client
        except Exception as docker_error:
            st.error("Could not connect to any Qdrant instance")
            st.info("Please configure Qdrant Cloud credentials or ensure local Qdrant is running")
            return None

# Check if Ollama is available and has the required model
def check_ollama_model(base_url, model_name):
    try:
        # Check if Ollama service is running
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code != 200:
            return False, "Ollama service is not available"
        
        # Check if the model is available
        models = response.json().get("models", [])
        available_models = [model["name"] for model in models]
        
        if model_name not in available_models:
            return False, f"Model '{model_name}' not found. Available models: {', '.join(available_models)}"
        
        return True, "Model is available"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"

# Initialize embedding model with better error handling
@st.cache_resource
def get_embeddings():
    # Try to get Ollama URL from environment variable with fallbacks
    ollama_urls = [
        os.environ.get("OLLAMA_BASE_URL"),
        "http://ollama:11434",
        "http://localhost:11434"
    ]
    
    for url in ollama_urls:
        if not url:
            continue
        
        try:
            # Check if Ollama is available and has the embedding model
            model_available, message = check_ollama_model(url, "nomic-embed-text")
            if not model_available:
                st.warning(f"Ollama at {url}: {message}")
                # Try to pull the model if it's not available
                try:
                    pull_response = requests.post(
                        f"{url}/api/pull",
                        json={"name": "nomic-embed-text"}
                    )
                    if pull_response.status_code == 200:
                        st.success(f"Successfully pulled nomic-embed-text model")
                        model_available = True
                    else:
                        st.error(f"Failed to pull model: {pull_response.text}")
                except Exception as pull_error:
                    st.error(f"Error pulling model: {str(pull_error)}")
                    continue
            
            if model_available:
                embeddings = OllamaEmbeddings(
                    model="nomic-embed-text",
                    base_url=url
                )
                # Test the embeddings
                test_embedding = embeddings.embed_query("test")
                if len(test_embedding) > 0:
                    st.success(f"Successfully connected to Ollama embeddings at {url}")
                    return embeddings
        except Exception as e:
            st.warning(f"Failed to connect to Ollama at {url}: {str(e)}")
            continue
    
    st.error("Could not connect to Ollama embedding service")
    st.info("Please ensure Ollama is running with the nomic-embed-text model")
    return None

# Initialize LLM with better error handling
@st.cache_resource
def get_llm():
    # Try to get Ollama URL from environment variable with fallbacks
    ollama_urls = [
        os.environ.get("OLLAMA_BASE_URL"),
        "http://ollama:11434",
        "http://localhost:11434"
    ]
    
    for url in ollama_urls:
        if not url:
            continue
        
        try:
            # Check if Ollama is available
            response = requests.get(f"{url}/api/tags")
            if response.status_code != 200:
                st.warning(f"Ollama service at {url} is not available")
                continue
                
            llm = ChatOllama(
                base_url=url,
                model="mistral",
                temperature=0.1
            )
            st.success(f"Successfully connected to Ollama LLM at {url}")
            return llm
        except Exception as e:
            st.warning(f"Failed to connect to Ollama LLM at {url}: {str(e)}")
            continue
    
    st.error("Could not connect to Ollama LLM service")
    st.info("Please ensure Ollama is running with the mistral model")
    return None

def process_document(file, progress_bar=None, status_text=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.getvalue())
        temp_path = temp_file.name
    
    try:
        # Create a temporary directory for splitting the PDF if needed
        with tempfile.TemporaryDirectory() as temp_dir:
            # First, get page count to estimate workload
            loader = PyPDFLoader(temp_path)
            
            # For very large PDFs, we'll process in batches
            status_text.text("Counting pages...")
            documents = loader.load()
            total_pages = len(documents)
            
            if progress_bar is not None:
                progress_bar.progress(5/100)  # Show initial progress
            
            status_text.text(f"Processing {total_pages} pages...")
            
            # Use a smaller chunk size for very large documents
            chunk_size = 500 if total_pages > 500 else 1000
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=50
            )
            
            # Process in batches for very large documents
            if total_pages > 200:
                all_chunks = []
                batch_size = 50  # Process 50 pages at a time
                num_batches = math.ceil(total_pages / batch_size)
                
                for i in range(0, total_pages, batch_size):
                    end_idx = min(i + batch_size, total_pages)
                    batch_docs = documents[i:end_idx]
                    batch_chunks = text_splitter.split_documents(batch_docs)
                    all_chunks.extend(batch_chunks)
                    
                    if progress_bar is not None:
                        # Update progress (5-70% range for document processing)
                        progress_percent = 5 + (65 * (end_idx / total_pages))
                        progress_bar.progress(int(progress_percent)/100)
                        status_text.text(f"Processed pages {i+1}-{end_idx} of {total_pages}...")
                
                chunks = all_chunks
            else:
                chunks = text_splitter.split_documents(documents)
                if progress_bar is not None:
                    progress_bar.progress(70/100)
            
            status_text.text(f"Created {len(chunks)} chunks from {total_pages} pages. Embedding chunks...")
            
            client = get_qdrant_client()
            if client is None:
                status_text.text("Error: Qdrant client not available. Please configure Qdrant connection.")
                return None, 0, 0
                
            embeddings = get_embeddings()
            if embeddings is None:
                status_text.text("Error: Embedding model not available. Please configure Ollama connection.")
                return None, 0, 0
            
            collection_name = file.name.replace('.pdf', '').replace(' ', '_').lower()
            
            # Create or recreate collection
            try:
                if client.collection_exists(collection_name):
                    status_text.text(f"Collection {collection_name} already exists. Recreating...")
                    client.delete_collection(collection_name)
                
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=768,
                        distance=models.Distance.COSINE
                    )
                )
            except Exception as e:
                status_text.text(f"Error creating collection: {str(e)}")
                return None, 0, 0
            
            # For large chunk sets, batch the embeddings
            vectorstore = Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings
            )
            
            # Process embeddings in batches for large documents
            if len(chunks) > 100:
                batch_size = 50  # Process 50 chunks at a time
                num_batches = math.ceil(len(chunks) / batch_size)
                
                for i in range(0, len(chunks), batch_size):
                    end_idx = min(i + batch_size, len(chunks))
                    batch = chunks[i:end_idx]
                    vectorstore.add_documents(batch)
                    
                    if progress_bar is not None:
                        # Update progress (70-95% range for embedding)
                        progress_percent = 70 + (25 * (end_idx / len(chunks)))
                        progress_bar.progress(int(progress_percent)/100)
                        status_text.text(f"Embedded chunks {i+1}-{end_idx} of {len(chunks)}...")
                        time.sleep(0.1)  # Small delay to prevent UI freezing
            else:
                vectorstore.add_documents(chunks)
                if progress_bar is not None:
                    progress_bar.progress(95/100)
            
            if progress_bar is not None:
                progress_bar.progress(100/100)
                status_text.text("Processing complete!")
                
            return collection_name, len(chunks), total_pages
    
    finally:
        os.unlink(temp_path)

# Sidebar for uploading documents
with st.sidebar:
    st.header("Document Processing")
    
    # Ollama configuration section
    st.subheader("Ollama Configuration")
    
    # Show current configuration
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "")
    if ollama_url:
        st.success(f"Ollama URL configured: {ollama_url}")
    else:
        st.warning("Ollama URL not configured")
        
    # Allow setting configuration in the UI
    with st.expander("Configure Ollama"):
        ollama_url_input = st.text_input("Ollama URL", value=ollama_url or "http://localhost:11434")
        
        if st.button("Save Ollama Configuration"):
            # In Streamlit Cloud, we can't modify environment variables directly
            # This is a workaround to store them in session state
            st.session_state.ollama_url = ollama_url_input
            
            # Update the environment variables for the current session
            os.environ["OLLAMA_BASE_URL"] = ollama_url_input
            
            # Clear the cached client to force recreation
            st.cache_resource.clear()
            st.success("Ollama configuration updated. Reconnecting...")
            st.experimental_rerun()
    
    # Qdrant configuration section
    st.subheader("Qdrant Configuration")
    
    # Show current configuration
    qdrant_url = os.environ.get("QDRANT_URL", "")
    if qdrant_url:
        st.success("Qdrant Cloud configured")
    else:
        st.warning("Qdrant Cloud not configured")
        
    # Allow setting configuration in the UI
    with st.expander("Configure Qdrant Cloud"):
        qdrant_url_input = st.text_input("Qdrant URL", value=qdrant_url)
        qdrant_api_key_input = st.text_input("Qdrant API Key", type="password", 
                                           value=os.environ.get("QDRANT_API_KEY", ""))
        
        if st.button("Save Qdrant Configuration"):
            # In Streamlit Cloud, we can't modify environment variables directly
            # This is a workaround to store them in session state
            st.session_state.qdrant_url = qdrant_url_input
            st.session_state.qdrant_api_key = qdrant_api_key_input
            
            # Update the environment variables for the current session
            os.environ["QDRANT_URL"] = qdrant_url_input
            os.environ["QDRANT_API_KEY"] = qdrant_api_key_input
            
            # Clear the cached client to force recreation
            st.cache_resource.clear()
            st.success("Qdrant configuration updated. Reconnecting...")
            st.experimental_rerun()
    
    # Collection management section
    st.subheader("Collections")
    client = get_qdrant_client()
    
    collection_names = []
    if client is not None:
        try:
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections] if collections else []
        except Exception as e:
            st.error(f"Error fetching collections: {str(e)}")
    
    if collection_names:
        selected_collection = st.selectbox(
            "Select an existing collection", 
            options=collection_names,
            index=0 if "collection_name" not in st.session_state else 
                  collection_names.index(st.session_state.collection_name) if 
                  st.session_state.collection_name in collection_names else 0
        )
        
        if st.button("Use Selected Collection"):
            st.session_state.collection_name = selected_collection
            st.success(f"Now using collection: {selected_collection}")
    else:
        st.info("No collections available. Upload a document to create one.")
    
    # Document upload section
    st.subheader("Upload New Document")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024 * 1024):.2f} MB"
        }
        st.write(file_details)
        
        if st.button("Process Document"):
            if client is None:
                st.error("Qdrant client not available. Please configure Qdrant connection first.")
            else:
                # Create a progress bar and status text
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    with st.spinner("Processing document..."):
                        collection_name, num_chunks, total_pages = process_document(
                            uploaded_file, 
                            progress_bar=progress_bar,
                            status_text=status_text
                        )
                        
                        if collection_name:
                            st.session_state.collection_name = collection_name
                            st.success(f"Document processed into {num_chunks} chunks from {total_pages} pages and stored in collection: {collection_name}")
                        else:
                            st.error("Failed to process document")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                finally:
                    # Ensure progress bar completes even if there's an error
                    progress_bar.progress(100/100)

# Main chat interface
st.subheader("Ask questions about your document")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    client = get_qdrant_client()
    embeddings = get_embeddings()
    llm = get_llm()
    
    # Check if all required components are available
    if client is None:
        with st.chat_message("assistant"):
            st.error("Qdrant client not available. Please configure Qdrant connection.")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I can't answer your question because the Qdrant database is not available. Please configure the Qdrant connection in the sidebar."})
    elif embeddings is None:
        with st.chat_message("assistant"):
            st.error("Embedding model not available. Please configure Ollama connection.")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I can't answer your question because the embedding model is not available. Please configure the Ollama connection in the sidebar and ensure the nomic-embed-text model is installed."})
    elif llm is None:
        with st.chat_message("assistant"):
            st.error("LLM not available. Please configure Ollama connection.")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I can't answer your question because the language model is not available. Please