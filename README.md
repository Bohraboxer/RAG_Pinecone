# AI Question Answering System

This project provides an AI-based system that answers questions from a PDF document. It uses OpenAI's gpt-4o-mini-2024-07-18 model for natural language processing and Pinecone for storing and retrieving document embeddings.

## Features
- Upload a PDF file and ask multiple questions about the content.
- Uses Pinecone for document indexing and querying.
- OpenAI’s gpt-4o-mini-2024-07-18 processes the context and generates answers based on document contents.
- Supports automatic creation of Pinecone index if it doesn't exist.

## Installation

Follow these steps to set up and run the project locally.

### Prerequisites
- Python 3.9 or higher
- Install dependencies using `pip`:
  
  ```bash
  pip install -r requirements.txt


API Usage
You can use Postman to interact with the API. Here's how to send a POST request in Postman:

Endpoint: POST /bulk_query
Set the method to POST.

Set the URL to http://127.0.0.1:5000/bulk_query.

Under the Body tab:

Choose form-data.
Add a key file with type File and select the PDF file you want to upload.
Add a key questions[] with type Text for each question you want to ask.
Example:

Key: file, Value: (Choose your PDF file)
Key: questions[], Value: "What is the document about?"
Key: questions[], Value: "Who is the author?"
Press Send to get the response. The server will return the answers to the provided questions in a JSON format.

