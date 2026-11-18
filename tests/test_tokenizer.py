import pytest
from nebula.core.tokenizer import NebulaTokenizer


def test_tokenizer_encode_decode():
    tok = NebulaTokenizer()
    text = "Hello Nebula!"
    ids = tok.encode(text, add_bos=False)
    decoded = tok.decode(ids)
    assert decoded == text


def test_tokenizer_special_tokens():
    tok = NebulaTokenizer()
    ids = tok.encode("Test", add_bos=True, add_eos=True)
    assert ids[0] == tok.BOS_TOKEN_ID
    assert ids[-1] == tok.EOS_TOKEN_ID


def test_tokenizer_analysis():
    tok = NebulaTokenizer()
    metadata = tok.analyze("ABC")
    assert len(metadata) == 4  # BOS + 3 chars
    assert metadata[1].text == "A"
    assert metadata[1].token_id == 65
