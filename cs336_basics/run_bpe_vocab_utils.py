from tests.common import gpt2_bytes_to_unicode
import json

def load_merges_gpt2(path):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(path, encoding="utf-8") as f:
        merges = [tuple(line.rstrip().split(" ")) for line in f]
        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in merges
        ]
    return merges

def load_vocab_gpt2(path):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
        vocab = {
            int(gpt2_vocab_index): bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_index, gpt2_vocab_item in vocab.items()
        }
    return vocab

vocab = load_vocab_gpt2("vocab_tinystories.json")
merges = load_merges_gpt2("merges_tinystories.txt")
print(vocab)
# print(merges)
print(sorted(vocab.values(), key=lambda x: -len(x))[0].decode("utf-8", errors="replace"))