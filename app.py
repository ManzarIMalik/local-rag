import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import textwrap
import time

st.set_page_config(
    page_title="RAGify Your Data", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Load the document
loader = PyMuPDFLoader('./data/380455eng.pdf')
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter()
docs = text_splitter.split_documents(documents)

# Initialize embeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text')

#Save and Load the FAISS database
db = FAISS.from_documents(docs, embeddings)
db.save_local('./faiss_db/', index_name='usesco_ai')
db = FAISS.load_local('./faiss_db/', embeddings=embeddings, index_name='usesco_ai', allow_dangerous_deserialization=True)

# Initialize retriever
retriever = db.as_retriever(search_kwargs={"k": 10, "search_type": "similarity"})

# Define the prompt template
prompt_template = """
You need either to explain the concept or answer the question about Computer Vision. 
Be detailed, use simple words and examples in your explanations. If required, utilize the relevant information.
Also give source of information, along with page number which relates to retrieved content.

{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Function to wrap text while preserving newlines
def wrap_text_preserve_newlines(text, width=70):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Function to process LLM response
def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])
    sources_used = '\n'.join(
        [
            source.metadata['title']
            + ' - page: '
            + str(source.metadata['page']+1)
            for source in llm_response['source_documents']
        ]
    )
    ans = f"{ans}\n\nSources:\n{sources_used}"
    return ans

# Function to get LLM answer with streaming
def llm_ans(query, model, response_placeholder):
    llm = Ollama(model=model, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False
    )

    start = time.time()
    llm_response = qa_chain.invoke(query)

    response_text = ""
    # Stream the response
    for chunk in llm_response['result'].split('. '):  # Assuming chunks are split by sentences
        response_text += chunk + '. '
        wrapped_response_text = wrap_text_preserve_newlines(response_text)
        response_placeholder.text(wrapped_response_text)
        time.sleep(0.1)  # Simulate delay for streaming effect

    # Process the final response to include sources and time elapsed
    ans = process_llm_response(llm_response)
    end = time.time()
    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    response_placeholder.text(ans + time_elapsed_str)

# Streamlit app
st.title("Your Data - Ask Anything!")

# Model selection
model = st.selectbox("Choose LLM model", ["llama3.2:latest", "phi3.5:latest", "llama3.1:latest"])

# Query input
query = st.text_input("Type your query", on_change=lambda: st.session_state.update({"execute": True}))

# Execute query on enter or button click
if st.button("Get Answer") or st.session_state.get("execute", False):
    st.session_state["execute"] = False
    if query:
        response_placeholder = st.empty()  # Placeholder for streaming response
        st.write("Processing your query...")
        llm_ans(query, model, response_placeholder)
        st.empty()  # Clear the processing message
    else:
        st.write("Please enter a query.")