# README

This Streamlit app allows users to chat with their documents using a conversational AI model. You can upload PDFs, DOCX, TXT, and MD files, or provide URLs, and the app will process the data for interactive Q&A. It integrates FAISS for efficient document retrieval and uses LLMs (like Ollama) for generating responses.

## Features

- **Upload Documents**: Supports PDF, DOCX, TXT, and MD files.
- **URL Support**: Load content from web pages.
- **FAISS Indexing**: Efficient retrieval of document data.
- **LLM Chat**: Ask questions about your documents and get AI-generated responses.
- **Chat History**: View previous conversations.
- **Manage Data**: Clear the vector store and chat history.
- **Model Selection**: Choose from multiple LLMs.

## Installation & Setup

1. **Clone the repository and install dependencies**:

   ```bash
   git clone https://github.com/your-repo/chat-with-your-data.git
   cd chat-with-your-data
   pip install -r requirements.txt
   ```

2. **Start the LLM server and run the app**:

   ```bash
   # Start the LLM server (e.g., Ollama)
   ollama serve

   # In a new terminal, run the Streamlit app
   streamlit run app.py
   ```

## Usage

1. **Upload Files** or add **URLs** via the sidebar.
2. **Ask Questions** about the content; the app will retrieve and display relevant information.
3. **Clear History** and manage the vector store using sidebar options.


## Contributing

We welcome contributions from the community! If you would like to contribute to this project.

## License

This project is licensed under the [MIT License](LICENSE).


