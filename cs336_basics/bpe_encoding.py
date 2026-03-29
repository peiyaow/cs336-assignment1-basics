'''
This is the BPE encoding example in 2.6 BPE encoding example
'''
import regex as re
from bpe_utils import PAT

corpus = 'the cat ate'
vocab = {
    0: b' ', 
    1: b'a', 
    2: b'c', 
    3: b'e', 
    4: b'h', 
    5: b't', 
    6: b'th', 
    7: b' c', 
    8: b' a', 
    9: b'the', 
    10: b' at'
}
merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]

def find_merges(bytes, merges):
    i = 0
    res = ()
    found = False
    while i < len(bytes):
        if i < len(bytes) - 1 and bytes[i:(i+2)] in merges:
            res += (bytes[i] + bytes[i+1], )
            i += 2
            found = True
        else:
            res += (bytes[i],)
            i += 1
    return found, res

inverse_vocab = {v:k for k, v in vocab.items()}

token_ids = []
for m in re.finditer(PAT, corpus):
    cur_pre_token = m.group() 
    cur_bytes = tuple(bytes([b]) for b in cur_pre_token.encode("utf-8")) 
    cur_token_ids = []
    while True:
        found, cur_bytes = find_merges(cur_bytes, merges)
        if not found:
            break
    for byte in cur_bytes:
        cur_token_ids.append(inverse_vocab[byte])
    
    token_ids += cur_token_ids

print(token_ids)