from flask import Flask, request, jsonify
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import CTransformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load .env vars
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot"

# Init Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check index
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Index '{INDEX_NAME}' not found.")

# ✅ Get index using Pinecone v3 client
index = pc.Index(INDEX_NAME)

# ✅ TEMP PATCH: Create fake class to trick LangChain into thinking it's old Index class
import types
import pinecone as pinecone_module
pinecone_module.Index = type(index)  # Monkey patch the module

# ✅ Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Connect to vector store using patched Index
docsearch = LangchainPinecone(index=index, embedding=embeddings, text_key="text")

# ✅ Load LLM
llm = CTransformers(
    model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={"max_new_tokens": 512, "temperature": 0.7}
)

# Load QA Chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Flask app
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    docs = docsearch.similarity_search(question, k=3)
    answer = qa_chain.run(input_documents=docs, question=question)

    return jsonify({
        "question": question,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(debug=True)
