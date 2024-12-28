from flask import Flask, request, jsonify
from services.pdf_reader import updating_data2db
from dotenv import load_dotenv
from pinecone import Pinecone
import os
from services.embeddings import embeddings_for_each_chunk
from services.query_processing import formatting_query
from services.llm import llm_call

load_dotenv()
api_key = os.environ.get("PINECONE_API_KEY")

app = Flask(__name__)

# API Endpoint
@app.route('/bulk_query', methods=['POST'])
def bulk_query():
    try:
        # Extract file and questions from the request
        file = request.files['file']
        questions = request.form.getlist('questions[]')

        # Save uploaded file temporarily
        file_path = f"./temp/{file.filename}"
        os.makedirs("./temp", exist_ok=True)
        file.save(file_path)

        # Parameters for Pinecone and document processing
        chunk_size = 4
        overlap = 1
        index_name = "zania-pc"
        top_k = 3

        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            # If index does not exist, create and upload data
            updating_data2db(file_path, chunk_size=chunk_size, overlap=overlap, index_name=index_name)
        else:
            print(f"Index '{index_name}' already exists. Skipping data upload.")

        # Process each question
        answers = {}
        for question in questions:
            embedded_query = embeddings_for_each_chunk(question)
            input_to_llm = formatting_query(query=question, embedded_query=embedded_query, index_name=index_name, top_k=top_k)
            answer = llm_call(input_to_llm)
            answers[question] = answer

        # Cleanup temporary file
        os.remove(file_path)

        # Return answers in JSON format
        return jsonify({"answers": answers})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
