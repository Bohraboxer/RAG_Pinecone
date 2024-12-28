from services.pdf_reader import updating_data2db
from dotenv import load_dotenv
from pinecone import Pinecone
import os
from services.embeddings import embeddings_for_each_chunk
from services.query_processing import formatting_query
from services.llm import llm_call

load_dotenv()
api_key = os.environ.get("PINECONE_API_KEY")


data_path = "data/handbook.pdf"
chunk_size = 4
overlap = 1
index_name = "zania-pc"

query = "What is the document about?"
embedded_query = embeddings_for_each_chunk(query)



# Initialize Pinecone
pc = Pinecone(api_key=api_key)
# Connect to your index
index = pc.Index(index_name)
# Describe index stats
index_stats = index.describe_index_stats()
# Extract and print the number of records
num_records = index_stats.get("total_vector_count", 0)
if num_records == 0:
    updating_data2db(data_path, chunk_size=chunk_size, overlap=overlap, index_name=index_name)
else:
    # Call ask_query to start interacting with the Pinecone database
    input_to_llm = formatting_query(query=query, embedded_query=embedded_query, index_name=index_name, top_k=3)
    print(input_to_llm)
    output = llm_call(input_to_llm)
    print(output)

