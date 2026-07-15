import os
from typing import Any

from dotenv import load_dotenv
from google import genai

from config import GEMINI_MODEL

load_dotenv()


def _get_client() -> Any:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    return genai.Client(api_key=gemini_api_key)


def _extract_text(chunk: Any) -> str | None:
    if chunk is None:
        return None

    if isinstance(chunk, str):
        return chunk

    if hasattr(chunk, "text") and chunk.text:
        return chunk.text

    if isinstance(chunk, tuple):
        for item in chunk:
            text = _extract_text(item)
            if text:
                return text
        return None

    if isinstance(chunk, list):
        for item in chunk:
            text = _extract_text(item)
            if text:
                return text
        return None

    if hasattr(chunk, "candidates"):
        parts = []
        for candidate in getattr(chunk, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                text = _extract_text(part)
                if text:
                    parts.append(text)
        if parts:
            return "".join(parts)

    if hasattr(chunk, "parts"):
        parts = []
        for part in getattr(chunk, "parts", []) or []:
            text = _extract_text(part)
            if text:
                parts.append(text)
        if parts:
            return "".join(parts)

    return None


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

    try:
        client = _get_client()

        if hasattr(client.models, "generate_content_stream"):
            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        else:
            stream = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

        for chunk in stream:
            text = _extract_text(chunk)
            if text:
                yield text

    except Exception as exc:
        yield f"I could not generate a response. Error: {exc}"
