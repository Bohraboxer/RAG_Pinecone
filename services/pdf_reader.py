from PyPDF2 import PdfReader
import re
import os
from pinecone import Pinecone, ServerlessSpec
from .embeddings import embeddings_for_each_chunk
from dotenv import load_dotenv


load_dotenv()


def reading_files(data_path: str) -> str:
    # reading pdf file
    storing_info = ""
    reader = PdfReader(data_path)
    number_of_pages = len(reader.pages)
    for page in range(number_of_pages):
        pages = reader.pages[page]
        text = pages.extract_text()
        storing_info += text
    return storing_info
    
def chunk_text_by_sentence(text: str, chunk_size: int, overlap: int) -> list:
    """
    Splits text into chunks based on sentences with a specified overlap.

    Args:
        text (str): The input text to split.
        chunk_size (int): The number of sentences per chunk.
        overlap (int): The number of overlapping sentences between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than the chunk size.")
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    start = 0

    while start < len(sentences):
        end = start + chunk_size
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # Move forward while keeping overlap

    return chunks

def chunk2pinecone(chunks: list, index_name) -> str:

    api_key = os.environ.get("PINECONE_API_KEY")
    # configure client
    pc = Pinecone(api_key=api_key)
    
    cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
    region = os.environ.get('PINECONE_REGION') or 'us-east-1'
    spec = ServerlessSpec(cloud=cloud, region=region)
    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pc.list_indexes().names():
    # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=1536,  # dimensionality of text-embedding-ada-002
            metric='cosine',
            spec=spec
        )
    # connect to index
    index = pc.Index(index_name)
    print("Uploading Data.....")
    for num, chunk in zip(range(len(chunks)), chunks):
        embedded_chunk = embeddings_for_each_chunk(chunk)
        embedded_chunk_formatted = [{"id": f'chunk_num_{num}', "values": list(embedded_chunk), "metadata": {"text": chunk}}]
  
        index.upsert(embedded_chunk_formatted)

    return "Successfully transfered data to Pinecone database!!!"



def updating_data2db(data_path, chunk_size, overlap, index_name):
    information = reading_files(data_path)
    chunks = chunk_text_by_sentence(information, chunk_size, overlap)
    chunk2pinecone(chunks, index_name)
    return "Successfully Embedded and Transfered data to Database...!!!"


