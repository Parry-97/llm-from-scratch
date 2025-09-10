import re
import pytest


@pytest.fixture
def raw_text():
    return "Hello, world! This is a test. -- Is it?"


def test_text_splitting_and_vocab(raw_text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    assert len(preprocessed) == 13

    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)

    assert vocab_size == 13

    vocab = {word: index for index, word in enumerate(all_words)}

    assert vocab["Hello"] == 5
    assert vocab["world"] == 12
