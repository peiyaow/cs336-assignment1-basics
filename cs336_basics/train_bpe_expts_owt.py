from cs336_basics.bpe_utils import *

vocab, merges = run_train_bpe("../data/owt_train.txt", 32000, ["<|endoftext|>"], 12)

save_vocab_gpt2(vocab, "vocab_owt.json")
save_merges_gpt2(merges, "merges_owt.txt")

# slow version; too slow didnt finish
# create_pre_tokens took 180.0809s
# Merged 1000 tokens. Time taken 18548.2027s

# fast version
# create_pre_tokens took 175.4413s
# Merged 1000 tokens. Time taken 296.6841s
# Merged 2000 tokens. Time taken 376.4780s
# Merged 3000 tokens. Time taken 477.1144s
# Merged 4000 tokens. Time taken 615.2356s
# Merged 5000 tokens. Time taken 783.1639s
# Merged 6000 tokens. Time taken 982.1829s
# Merged 7000 tokens. Time taken 1208.5661s
# Merged 8000 tokens. Time taken 1454.1745s
# Merged 9000 tokens. Time taken 1723.3967s
# Merged 10000 tokens. Time taken 2016.3875s
# Merged 11000 tokens. Time taken 2334.1480s
# Merged 12000 tokens. Time taken 2675.6427s
# Merged 13000 tokens. Time taken 3039.3527s
# Merged 14000 tokens. Time taken 3422.1802s
# Merged 15000 tokens. Time taken 3824.5410s
# Merged 16000 tokens. Time taken 4248.5160s
# Merged 17000 tokens. Time taken 4690.6161s
# Merged 18000 tokens. Time taken 5151.1369s
# Merged 19000 tokens. Time taken 5630.0242s
# Merged 20000 tokens. Time taken 6126.8216s
# Merged 21000 tokens. Time taken 6639.4287s
# Merged 22000 tokens. Time taken 7172.1333s
# Merged 23000 tokens. Time taken 7726.2503s
# Merged 24000 tokens. Time taken 8295.8542s
# Merged 25000 tokens. Time taken 8881.8961s
# Merged 26000 tokens. Time taken 9483.3014s
# Merged 27000 tokens. Time taken 10099.4212s
# Merged 28000 tokens. Time taken 10729.4726s
# Merged 29000 tokens. Time taken 11372.7777s
# Merged 30000 tokens. Time taken 12030.3340s
# Merged 31000 tokens. Time taken 12699.1108s
# run_train_bpe_on_pre_tokens took 13207.5314s