from langchain.document_loaders import PyPDFLoader, DirectoryLoader

def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents


from langchain.text_splitter import RecursiveCharacterTextSplitter

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

from langchain_huggingface import HuggingFaceEmbeddings

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings