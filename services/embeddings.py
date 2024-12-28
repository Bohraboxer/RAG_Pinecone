import openai


def embeddings_for_each_chunk(text: str) -> list:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
        )
    result = response["data"][0]["embedding"] 
    return result