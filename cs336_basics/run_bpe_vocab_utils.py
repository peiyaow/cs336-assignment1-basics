from bpe_utils import load_vocab_gpt2, load_merges_gpt2

vocab = load_vocab_gpt2("vocab_tinystories.json")
merges = load_merges_gpt2("merges_tinystories.txt")
print("Tinystories longest token:")
print(sorted(vocab.values(), key=lambda x: -len(x))[0])
# b' accomplishment'

vocab = load_vocab_gpt2("vocab_owt.json")
merges = load_merges_gpt2("merges_owt.txt")
# print(vocab)
print("OWT longest token:")
print(sorted(vocab.values(), key=lambda x: -len(x))[0])
# b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'