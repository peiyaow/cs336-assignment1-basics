from cs336_basics.bpe_utils import *

vocab, merges = run_train_bpe("../data/owt_train.txt", 32000, ["<|endoftext|>"], 12)

save_vocab_gpt2(vocab, "vocab_owt.json")
save_merges_gpt2(merges, "merges_owt.txt")

# create_pre_tokens took 180.0809s
# Merged 1000 tokens. Time taken 18548.2027s