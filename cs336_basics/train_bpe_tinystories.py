from cs336_basics.bpe_utils import *

vocab, merges = run_train_bpe("../data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"], 4)

save_vocab_gpt2(vocab, "vocab_tinystories.json")
save_merges_gpt2(merges, "merges_tinystories.txt")

# slow version
# create_pre_tokens took 57.2225s
# Merged 1000 tokens. Time taken 57.9865s
# Merged 2000 tokens. Time taken 106.3619s
# Merged 3000 tokens. Time taken 150.9018s
# Merged 4000 tokens. Time taken 193.9903s
# Merged 5000 tokens. Time taken 235.1201s
# Merged 6000 tokens. Time taken 274.6158s
# Merged 7000 tokens. Time taken 314.8250s
# Merged 8000 tokens. Time taken 353.4530s
# Merged 9000 tokens. Time taken 391.2095s
# run_train_bpe_on_pre_tokens took 417.8975s

# fast version
# create_pre_tokens took 56.0175s
# Merged 1000 tokens. Time taken 2.3699s
# Merged 2000 tokens. Time taken 4.6959s
# Merged 3000 tokens. Time taken 7.6005s
# Merged 4000 tokens. Time taken 10.8074s
# Merged 5000 tokens. Time taken 14.2475s
# Merged 6000 tokens. Time taken 17.9317s
# Merged 7000 tokens. Time taken 21.8772s
# Merged 8000 tokens. Time taken 26.1207s
# Merged 9000 tokens. Time taken 30.5022s
# run_train_bpe_on_pre_tokens took 33.7094s