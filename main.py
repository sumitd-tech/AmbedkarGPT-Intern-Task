from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# --- Load your document ---
loader = TextLoader("speech.txt", encoding="utf-8")
docs = loader.load()

# --- Split document into chunks ---
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# --- Create embeddings and vector DB ---
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(
    chunks, 
    embedding=emb, 
    persist_directory="chroma_db"
)
vectordb.persist()

# --- Use mini Ollama model ---
llm = Ollama(model="llama3.2:3b")

# --- Friendly & polite prompt template ---
prompt_template = """
You are a friendly and helpful AI assistant. Use ONLY the context below to answer the user's question.
Respond politely and in a conversational tone. If the answer is not in the context, say:
"I'm sorry, but I don’t know the answer based on the provided text."

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# --- Set up RetrievalQA chain ---
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# --- Chat loop ---
print("AmbedkarGPT — ask anything (type 'exit' or 'quit' to stop)")

while True:
    q = input("\nQuestion: ")
    if q.lower() in ("exit", "quit"):
        break

    result = qa.invoke({"query": q})
    print("\nAnswer:\n", result["result"])
