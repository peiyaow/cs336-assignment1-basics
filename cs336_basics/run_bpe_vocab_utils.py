from bpe_utils import load_vocab_gpt2, load_merges_gpt2

vocab = load_vocab_gpt2("vocab_tinystories.json")
merges = load_merges_gpt2("merges_tinystories.txt")
print(vocab)
# print(merges)
print(sorted(vocab.values(), key=lambda x: -len(x))[0].decode("utf-8", errors="replace"))