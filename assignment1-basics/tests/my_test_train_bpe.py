import json
import time

from adapters import run_train_bpe
from common import FIXTURES_PATH, gpt2_bytes_to_unicode

def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.example"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    print(merges)
    expected = [(b's', b't'), (b'e', b'st'), (b'o', b'w'), (b'l', b'ow'), (b'w', b'est'), (b'n', b'e'), (b'ne', b'west'), (b' ', b'newest'), (b' ', b'low'), (b'w', b'i'), (b'wi', b'd'), (b'wid', b'est'), (b' ', b'widest'), (b'e', b'r'), (b'low', b'er'), (b' low', b'er')]
    for i, (merge, ref_merge) in enumerate(zip(merges, expected)):
        if merge != ref_merge:
            print(f"Merge mismatch at index {i}: got {merge}, expected {ref_merge}")
    assert merges == expected
    # 正确答案 [(b's', b't'), (b'e', b'st'), (b'o', b'w'), (b'l', b'ow'), (b'w', b'est'), (b'n', b'e'), (b'ne', b'west'), (b' ', b'newest'), (b' ', b'low'), (b'w', b'i'), (b'wi', b'd'), (b'wid', b'est'), (b' ', b'widest'), (b'e', b'r'), (b'low', b'er'), (b' low', b'er')]

if __name__ == "__main__":
    test_train_bpe()