import pytest

from core.embeddings.encoder import TextEncoder


def test_encoder_rejects_empty_text():
    encoder = TextEncoder.__new__(TextEncoder)

    with pytest.raises(ValueError):
        encoder.encode("")