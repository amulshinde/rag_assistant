def iter_response_chunks(response):
    """Yield text chunks from a streaming response, with a safe fallback."""
    try:
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk
    except Exception as exc:
        yield f"[Error] {exc}"
