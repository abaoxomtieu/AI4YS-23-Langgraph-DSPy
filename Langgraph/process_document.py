from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

vector_db_path = "vectorstores/db_faiss"
load_dotenv()


def create_db_from_files(folder_path="./data"):
    # Load all data in data folder
    loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    print("Success")
    return db


create_db_from_files()
