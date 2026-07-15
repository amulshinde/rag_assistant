import unittest

from frontend.streaming_utils import iter_response_chunks


class DummyResponse:
    def __init__(self, chunks=None, error=None):
        self._chunks = chunks or []
        self._error = error

    def iter_content(self, chunk_size=None, decode_unicode=True):
        if self._error is not None:
            raise self._error
        for chunk in self._chunks:
            yield chunk


class StreamingUtilsTests(unittest.TestCase):
    def test_iter_response_chunks_yields_streamed_content(self):
        response = DummyResponse(["Hello", " ", "world"])

        chunks = list(iter_response_chunks(response))

        self.assertEqual(chunks, ["Hello", " ", "world"])

    def test_iter_response_chunks_falls_back_on_stream_errors(self):
        response = DummyResponse(error=RuntimeError("stream broke"))

        chunks = list(iter_response_chunks(response))

        self.assertEqual(chunks, ["[Error] stream broke"])


if __name__ == "__main__":
    unittest.main()
