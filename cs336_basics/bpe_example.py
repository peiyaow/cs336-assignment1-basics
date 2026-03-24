'''
This is the stylized BPE example in 2.4 BPE Tokenizer Training
'''

import regex as re

corpus = '''
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
'''

vocab = {i for i in range(256)}
PAT = r"\S+" # split on whitespace only
vocab.add(b"<|endoftext|>")

pre_tokens = {}
for m in re.finditer(PAT, corpus):
    cur_bytes = m.group() 
    if tuple(cur_bytes) in pre_tokens:
        pre_tokens[tuple(cur_bytes)] += 1
    else:
        pre_tokens[tuple(cur_bytes)] = 1

def construct_merge_bytes_dict(pre_tokens):
    merge_bytes_count = {}
    for k, v in pre_tokens.items():
        for i in range(len(k)-1):
            if k[i:(i+2)] not in merge_bytes_count:
                merge_bytes_count[k[i:(i+2)]] = v
            else:
                merge_bytes_count[k[i:(i+2)]] += v
    return merge_bytes_count

def find_max_merge_bytes(merge_bytes_count):
    max_bytes = None
    max_count = 0
    for k, v in merge_bytes_count.items():
        if max_bytes is None:
            max_bytes = k
            max_count = v
        if max_count < v:
            max_bytes = k
            max_count = v
        if max_count == v and max_bytes < k:
            max_bytes = k
    return max_bytes

def merge_bytes_in_pre_tokens(pre_tokens, max_bytes):
    new_merged_pre_tokens = {}
    for k in pre_tokens:
        i = 0
        cur_seq = ()
        while i < len(k):
            if (i+2 <= len(k) and k[i:(i+2)] != max_bytes) or i+2 > len(k):
                cur_seq += (k[i],)
                i += 1
            else:
                cur_seq += (k[i]+k[i+1],)
                i += 2
        new_merged_pre_tokens[cur_seq] = pre_tokens[k]
    return new_merged_pre_tokens

MAX_ITER = 6
n_iter = 0
while n_iter < MAX_ITER:
    merge_bytes_count = construct_merge_bytes_dict(pre_tokens)
    max_bytes = find_max_merge_bytes(merge_bytes_count)
    pre_tokens = merge_bytes_in_pre_tokens(pre_tokens, max_bytes)
    print(max_bytes)
    vocab.add(max_bytes)
    n_iter += 1

print(vocab)


        
