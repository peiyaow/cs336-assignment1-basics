from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter
import multiprocessing as mp
import time
import json
from tests.common import gpt2_bytes_to_unicode
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def create_pre_tokens_from_doc(doc, re_pattern, pre_tokens):
    for m in re.finditer(re_pattern, doc):
        cur_bytes = tuple(m.group().encode("utf-8"))
        if cur_bytes in pre_tokens:
            pre_tokens[cur_bytes] += 1
        else:
            pre_tokens[cur_bytes] = 1
    return pre_tokens

def construct_local_pair_counts(pre_token):
    local_pair_counts = Counter()
    for i in range(len(pre_token)-1):
        local_pair_counts[pre_token[i:(i+2)]] += 1
    return local_pair_counts

def index_global_and_pre_token_level_pair_counts(pre_tokens):
    """Aggregate consecutive byte-pair statistics over pretokens and their corpus counts.

    Args:
        pre_tokens: Mapping from each pretoken (tuple of bytes) to how often it appears
            in the corpus.

    Returns:
        A tuple of:
        - global_pair_counts: Counter mapping each byte pair to its total count across
            the corpus (local pair counts weighted by pretoken frequency).
        - seq_pair_counts: Dict mapping each pretoken to a Counter of byte pairs
            within a single instance of that pretoken.
        - pair_to_pre_tokens: Dict mapping each byte pair to the set of pretokens
            that contain that pair.
    """
    global_pair_counts = Counter()
    seq_pair_counts = defaultdict(Counter)
    pair_to_pre_tokens = defaultdict(set)
    for pre_token in pre_tokens:
        cur_local_pair_counts = construct_local_pair_counts(pre_token)
        seq_pair_counts[pre_token] = cur_local_pair_counts
        for local_pair, count in cur_local_pair_counts.items():
            global_pair_counts[local_pair] += count*pre_tokens[pre_token]
            pair_to_pre_tokens[local_pair].add(pre_token)
    return global_pair_counts, seq_pair_counts, pair_to_pre_tokens

def index_byte_pairs_dict(pre_tokens):
    bytes2count = {}
    for k, v in pre_tokens.items():
        for i in range(len(k)-1):
            if k[i:(i+2)] not in bytes2count:
                bytes2count[k[i:(i+2)]] = v
            else:
                bytes2count[k[i:(i+2)]] += v
    return bytes2count

def find_max_merge_bytes(merge_bytes_count, vocab):
    max_pair = None
    max_count = -1

    for pair, count in merge_bytes_count.items():
        pair_bytes = (vocab[pair[0]], vocab[pair[1]])

        if max_pair is None:
            max_pair = pair
            max_count = count
            continue

        max_pair_bytes = (vocab[max_pair[0]], vocab[max_pair[1]])

        if count > max_count or (count == max_count and pair_bytes > max_pair_bytes):
            max_pair = pair
            max_count = count

    return max_pair

def merge_bytes_in_pre_tokens(pre_tokens, max_bytes, token_id):
    new_merged_pre_tokens = {}
    for k in pre_tokens:
        i = 0
        cur_seq = ()
        while i < len(k):
            if (i+2 <= len(k) and k[i:(i+2)] != max_bytes) or i+2 > len(k):
                cur_seq += (k[i],)
                i += 1
            else:
                cur_seq += (token_id,)
                i += 2
        new_merged_pre_tokens[cur_seq] = pre_tokens[k]
    return new_merged_pre_tokens

def merge_seq(seq, pair_to_merge, token_id):
    new_seq = ()
    i = 0
    while i < len(seq):
        if (i+2 <= len(seq) and seq[i:(i+2)] == pair_to_merge):
            new_seq += (token_id,)
            i += 2
        else:
            new_seq += (seq[i],)
            i += 1
    return new_seq

def create_pre_tokens(input_path, special_tokens, num_processes = 4):
    DOC_SPLIT_PAT = "|".join(re.escape(t) for t in special_tokens)
    pre_tokens = {}
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [
        (input_path, start, end, DOC_SPLIT_PAT) 
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]    
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, tasks)
    
    pre_tokens = Counter()
    for result in results:
        pre_tokens.update(result)
    
    return dict(pre_tokens)

def process_chunk(args):
    input_path, start, end, doc_split_pat = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    docs = re.split(doc_split_pat, chunk)
    pre_tokens = Counter()
    for doc in docs:
        if doc:
            for m in re.finditer(PAT, doc):
                cur_bytes = tuple(m.group().encode("utf-8"))
                pre_tokens[cur_bytes] += 1
    return pre_tokens

def run_train_bpe_on_pre_tokens(
    pre_tokens,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1
    
    token_count = len(vocab)
    merges = []
    
    start = time.perf_counter()
    while token_count < vocab_size:
        bytes2count = index_byte_pairs_dict(pre_tokens)
        max_byte_pairs = find_max_merge_bytes(bytes2count, vocab)
        pre_tokens = merge_bytes_in_pre_tokens(pre_tokens, max_byte_pairs, token_count)
        merges.append((vocab[max_byte_pairs[0]], vocab[max_byte_pairs[1]]))
        vocab[token_count] = vocab[max_byte_pairs[0]]+vocab[max_byte_pairs[1]]
        token_count += 1
        if (token_count - 256 - len(special_tokens))%1000 == 0:
            end = time.perf_counter()
            print(f"Merged {token_count - 256 - len(special_tokens)} tokens. Time taken {end - start:.4f}s")
    return vocab, merges

def run_train_bpe_on_pre_tokens_fast(pre_tokens,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1
    
    merges = []
    
    start = time.perf_counter()
    
    # fast version; iterate only on affected pre_tokens
    global_pair_counts, seq_pair_counts, pair_to_pre_tokens = \
        index_global_and_pre_token_level_pair_counts(pre_tokens)
    
    while next_token_id < vocab_size:
        max_pair = find_max_merge_bytes(global_pair_counts, vocab)
        vocab[next_token_id] = vocab[max_pair[0]]+vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
        affected_seqs = list(pair_to_pre_tokens[max_pair])
        for affected_seq in affected_seqs:
            seq_count = pre_tokens.pop(affected_seq)
            for pair, pair_count in seq_pair_counts[affected_seq].items():
                global_pair_counts[pair] -= seq_count*pair_count
                pair_to_pre_tokens[pair].remove(affected_seq)
            new_seq = merge_seq(affected_seq, max_pair, next_token_id)
            new_local_pair_counts = construct_local_pair_counts(new_seq)
            for pair, pair_count in new_local_pair_counts.items():
                global_pair_counts[pair] += seq_count*pair_count
                pair_to_pre_tokens[pair].add(new_seq)
            pre_tokens[new_seq] = seq_count
            del seq_pair_counts[affected_seq]
            seq_pair_counts[new_seq] = new_local_pair_counts
        del global_pair_counts[max_pair]
        next_token_id += 1
        if (next_token_id - 256 - len(special_tokens))%1000 == 0:
            end = time.perf_counter()
            print(f"Merged {next_token_id - 256 - len(special_tokens)} tokens. Time taken {end - start:.4f}s")
    
    return vocab, merges


def run_train_bpe(input_path, vocab_size, special_tokens, num_processes = 4, fast=True):
    start = time.perf_counter()
    pre_tokens = create_pre_tokens(input_path, special_tokens, num_processes)
    end = time.perf_counter()
    print(f"create_pre_tokens took {end - start:.4f}s")
    start = time.perf_counter()
    if fast:
        vocab, merges = run_train_bpe_on_pre_tokens_fast(pre_tokens, vocab_size, special_tokens)
    else:
        vocab, merges = run_train_bpe_on_pre_tokens(pre_tokens, vocab_size, special_tokens)
    end = time.perf_counter()
    print(f"run_train_bpe_on_pre_tokens took {end - start:.4f}s")
    return vocab, merges

def bytes_to_gpt2_str(b: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[byte] for byte in b)


def save_vocab_gpt2(vocab: dict[int, bytes], path: str) -> None:
    byte_encoder = gpt2_bytes_to_unicode()
    vocab_str = {str(k): bytes_to_gpt2_str(v, byte_encoder) for k, v in vocab.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, indent=2, ensure_ascii=False)
    
def save_merges_gpt2(merges: list[tuple[bytes, bytes]], path: str) -> None:
    byte_encoder = gpt2_bytes_to_unicode()
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            a_str = bytes_to_gpt2_str(a, byte_encoder)
            b_str = bytes_to_gpt2_str(b, byte_encoder)
            f.write(f"{a_str} {b_str}\n")