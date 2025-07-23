from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from dotenv import load_dotenv
import pinecone
import os

# Load environment variables from .env
load_dotenv()

# Pinecone credentials from env
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')  # optional

# Load and prepare documents
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone (new SDK)
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")  # or use env var

# Index config
index_name = "medical-chatbot"
dimension = 384  # for all-MiniLM-L6-v2

# Create index if not exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine"
    )
    print(f"✅ Created index '{index_name}'")
else:
    print(f"✅ Index '{index_name}' already exists")

# Connect to the index
index = pinecone.Index(index_name)

# Store the embeddings in Pinecone
try:
    docsearch = LangchainPinecone.from_texts(
        texts=[chunk.page_content for chunk in text_chunks],
        embedding=embeddings,
        index_name=index_name
    )
    print("🧠 Embeddings stored in Pinecone successfully!")
    print(f"✅ Total Chunks: {len(text_chunks)}")
    print(f"🔍 Sample Chunk: {text_chunks[0].page_content[:100]}")
except Exception as e:
    print("❌ Error uploading to Pinecone:", e)
