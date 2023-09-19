import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import bot_template, user_template, css
from langchain.llms import HuggingFaceHub
import os
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access the HuggingFace API
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def get_pdf_text(pdf_files):
    
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def get_chunk_text(text):

    tokens = word_tokenize(text)
    max_chunk_size = 100  # Maximum number of tokens per chunk
    chunks = [] 
    current_chunk = []
    current_chunk_length = 0

    # Iterate through the tokens
    for token in tokens:
        # If adding the token to the current chunk doesn't exceed the max_chunk_size
        if current_chunk_length + len(token) <= max_chunk_size:
            current_chunk.append(token)
            current_chunk_length += len(token)
        else:
            # If adding the token exceeds the limit, start a new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [token]
            current_chunk_length = len(token)

    # Add the last chunk (if any)
    if current_chunk:
        chunks.append(" ".join(current_chunk))


    return chunks

def get_vector_store(text_chunks):
    
    
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    
    return vectorstore