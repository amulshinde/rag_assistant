from google import genai
from config import GEMINI_MODEL

import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Pass the variable name GEMINI_API_KEY, not the string literal "GEMINI_API_KEY"
client = genai.Client(
    api_key=gemini_api_key
)

def ask_gemini(query, context):
    
    prompt = f"""
    You are a helpful assistant.

    Answer the user's question using ONLY the provided context.

    If the answer is not present in the context, say:
    "I could not find the answer in the provided document."

    Context: {context}

    Question: {query}

    Answer:
    """

    # Use client.models.generate_content and the 'contents' parameter
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    # Use response.text to extract the output string
    return response.text

# print(ask_gemini("What is the capital of France?", "France is a country in Europe. Its capital is Paris."))