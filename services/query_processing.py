from pinecone import Pinecone
import os
from dotenv import load_dotenv
from services.embeddings import embeddings_for_each_chunk


load_dotenv()
api_key = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Function to query Pinecone and get the most relevant chunks
def query_pinecone(embedded_query: list, index_name: str, top_k: int)->list:
    index = pc.Index(index_name)

    # Query Pinecone with the generated embedding
    results = index.query(
        vector=embedded_query,
        top_k=top_k,
        include_metadata=True  # Including metadata if available
    )
    
    # Retrieve the most relevant chunks and their metadata
    answers = []
    for match in results['matches']:
        answer_text = match['metadata'].get('text', 'No text available')
        answers.append(answer_text)
    
    return answers


# Example query function call
def ask_query(embedded_query: str, index_name: str, top_k: int) -> dict:
    answers = query_pinecone(embedded_query, index_name=index_name, top_k=top_k)
    answers_list = {}
    # Displaying the most relevant chunks as answers
    for idx, answer in enumerate(answers, 1):
        answers_list[idx]=answer
    return answers_list


def formatting_query(query: str, embedded_query: list, index_name: str, top_k: int)->str:
    context = ask_query(embedded_query=embedded_query, index_name=index_name, top_k=top_k)
    context_string = ""
    query_string = query
    for key, value in context.items():
        context_string += "ID: " + str(key) + "\n" + "Text: " + value + "\n\n"
    input_with_context = f"""
    Answer the following question based on the context provided below:
    
    Question: {query}
    
    Context: {context_string}"""

    return input_with_context