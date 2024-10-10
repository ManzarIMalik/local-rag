import os
import streamlit as st

from langchain_community.document_loaders import (
    TextLoader, 
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import textwrap
import time
import shutil  # Import shutil to delete directories

st.set_page_config(
    page_title="Chat with Your Data",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Functions ---

def wrap_text_preserve_newlines(text, width=70):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['answer'])
    sources_used = []
    for source in llm_response['source_documents']:
        source_name = source.metadata.get('source', 'Unknown')
        page = source.metadata.get('page')
        if isinstance(page, int):
            page_str = str(page + 1)
        else:
            page_str = 'N/A'
        sources_used.append(f"{source_name} - page: {page_str}")
    sources_used_str = '\n'.join(sources_used)
    ans = f"{ans}\n\nSources:\n{sources_used_str}"
    return ans

def llm_ans(chat_history, model, retriever, response_placeholder):
    # Initialize the LLM
    llm = Ollama(model=model, temperature=0.2, base_url="http://localhost:11434")
    
    # Create the ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )

    start = time.time()
    
    # Prepare chat history in the required format
    formatted_chat_history = []
    for i in range(0, len(chat_history) - 1, 2):
        user_message = chat_history[i]['content']
        assistant_message = chat_history[i+1]['content'] if i+1 < len(chat_history) else ''
        formatted_chat_history.append((user_message, assistant_message))
    
    # Run the chain to get the response
    llm_response = qa_chain.invoke({
        "question": chat_history[-1]['content'],
        "chat_history": formatted_chat_history
    })
    
    # Process and display the response
    ans = process_llm_response(llm_response)
    end = time.time()
    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    response_placeholder.markdown(ans + time_elapsed_str)
    
    # Append assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": llm_response['answer']})

def load_documents(uploaded_files, urls):
    import tempfile
    docs = []
    if uploaded_files:
        for doc_file in uploaded_files:
            if doc_file.type == "application/pdf":
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(doc_file.read())
                    tmp_file_path = tmp_file.name
                # Load the PDF using the file path
                loader = PyPDFLoader(tmp_file_path)
                docs.extend(loader.load())
                # Clean up the temporary file
                os.unlink(tmp_file_path)
            elif doc_file.name.endswith(".docx"):
                # Similar process for docx files
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                    tmp_file.write(doc_file.read())
                    tmp_file_path = tmp_file.name
                loader = Docx2txtLoader(tmp_file_path)
                docs.extend(loader.load())
                os.unlink(tmp_file_path)
            elif doc_file.type in ["text/plain", "text/markdown"]:
                # Similar process for text files
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                    tmp_file.write(doc_file.read())
                    tmp_file_path = tmp_file.name
                loader = TextLoader(tmp_file_path)
                docs.extend(loader.load())
                os.unlink(tmp_file_path)
            else:
                st.warning(f"Document type {doc_file.type} not supported.")
                continue

    if urls:
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading document from {url}: {e}")
    return docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

def create_vectorstore(docs, embeddings, reset=False):
    if reset:
        # Clear the existing vector store
        if os.path.exists('./faiss_db/'):
            shutil.rmtree('./faiss_db/')
    index_path = './faiss_db/user_data.faiss'
    if os.path.exists(index_path):
        db = FAISS.load_local(
            './faiss_db/',
            embeddings=embeddings,
            index_name='user_data',
            allow_dangerous_deserialization=True
        )
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local('./faiss_db/', index_name='user_data')
    return db

def clear_vectorstore():
    if os.path.exists('./faiss_db/'):
        shutil.rmtree('./faiss_db/')
        st.info("Vector store cleared.")
    else:
        st.warning("No vector store to clear.")

# --- Streamlit App ---

st.title("Chat with Your Data 💬")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for uploading documents and entering URLs
st.sidebar.header("Add Your Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (PDF, TXT, DOCX, MD)",
    type=["pdf", "txt", "docx", "md"],
    accept_multiple_files=True
)
urls_input = st.sidebar.text_area("Enter URLs (one per line)")
urls = urls_input.strip().split('\n') if urls_input else []

# Option to reset the vector store when loading new documents
reset_vectorstore = st.sidebar.checkbox("Reset Vector Store before Loading")

# Button to clear the vector store and chat history
st.sidebar.header("Manage Vector Store and Chat")
if st.sidebar.button("Clear Vector Store and Chat History"):
    clear_vectorstore()
    # Clear chat history
    st.session_state.chat_history = []
    st.info("Chat history cleared.")

# Load and process documents
if st.sidebar.button("Load Documents"):
    with st.spinner("Loading documents..."):
        documents = load_documents(uploaded_files, urls)
        if documents:
            docs = split_documents(documents)
            embeddings = OllamaEmbeddings(model='nomic-embed-text')
            db = create_vectorstore(docs, embeddings, reset=reset_vectorstore)
            st.success("Documents loaded and processed successfully.")
        else:
            st.warning("No valid documents to load.")

# Check if vector store exists
index_path = './faiss_db/user_data.faiss'
if os.path.exists(index_path):
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    db = FAISS.load_local(
        './faiss_db/',
        embeddings=embeddings,
        index_name='user_data',
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 10, "search_type": "similarity"})
else:
    st.warning("Please load documents to start chatting.")
    st.stop()

# Model selection
model = st.sidebar.selectbox("Choose LLM model", ["llama3.1:latest", "llama3.2:latest", "phi3.5:latest"])

# Display chat messages from history on the main page
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message"):
    # Append user's message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Placeholder for assistant's response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        st.write("Thinking...")
        llm_ans(st.session_state.chat_history, model, retriever, response_placeholder)