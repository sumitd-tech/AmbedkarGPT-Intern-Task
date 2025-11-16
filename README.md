# AmbedkarGPT-Intern-Task
This is a simple Q&A system based on a short speech by Dr. B.R. Ambedkar. It answers questions using only the text provided.

## Project Overview
The system demonstrates a Retrieval-Augmented Generation (RAG) pipeline using:

- LangChain :– To orchestrate the RAG pipeline.
- ChromaDB :– Local vector store for storing embeddings.
- HuggingFaceEmbeddings :– `sentence-transformers/all-MiniLM-L6-v2` for embedding the speech text.
- Ollama LLM :– Mini LLaMA model (`llama3.2:3b`) to generate answers.
- Text Splitter :– Splits the speech into manageable chunks for better retrieval.

The chatbot retrieves relevant context from the speech and generates polite, context-aware answers.

## How to Run
1. Create a virtual environment:
   python -m venv venv
   
2.Activate the environment:
- Windows:
  venv\Scripts\activate
  
- Linux/macOS:
  source venv/bin/activate
  
3.Install dependencies:
  pip install -r requirements.txt

4.Run the chatbot:
  python main.py
