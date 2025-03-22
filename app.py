import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import Qdrant
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient, models
import time
import math

# App title
st.title("📚 RAG with Ollama, Qdrant, and LangChain")

# Initialize Qdrant client
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        host=os.environ.get("QDRANT_HOST", "qdrant"),
        port=6333,
        prefer_grpc=True
    )

# Initialize embedding model
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
    )

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOllama(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"),
        model="mistral",
        temperature=0.1
    )

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
            embeddings = get_embeddings()
            
            collection_name = file.name.replace('.pdf', '').replace(' ', '_').lower()
            
            # Create or recreate collection
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
    
    # Collection management section
    st.subheader("Collections")
    client = get_qdrant_client()
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections] if collections else []
    
    if collection_names:
        selected_collection = st.selectbox(
            "Select an existing collection", 
            options=collection_names,
            index=0 if "collection_name" not in st.session_state else collection_names.index(st.session_state.collection_name) if st.session_state.collection_name in collection_names else 0
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
                    st.session_state.collection_name = collection_name
                    st.success(f"Document processed into {num_chunks} chunks from {total_pages} pages and stored in collection: {collection_name}")
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
    
    if "collection_name" in st.session_state:
        collection_name = st.session_state.collection_name
        vectorstore = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        # Increase k for large documents to get more context
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    else:
        retriever = None  # No retriever available when no document is uploaded
    
    with st.chat_message("assistant"):
        answer_container = st.empty()
        
        with st.spinner("Thinking..."):
            if retriever:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff",
                    return_source_documents=True
                )
                response = qa_chain({"query": prompt})
                answer = response["result"]
                
                # Optional: Display source documents in an expandable section
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(doc.page_content)
                        st.markdown("---")
            else:
                response = llm.invoke(prompt)
                # Extract just the content from the response
                if hasattr(response, 'content'):
                    answer = response.content
                else:
                    answer = str(response)
            
            answer_container.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
