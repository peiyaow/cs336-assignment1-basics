from cs336_basics.pretokenization_example import find_chunk_boundaries
from tests.adapters import index_byte_pairs_dict, find_max_merge_bytes, merge_bytes_in_pre_tokens, index_global_and_pre_token_level_pair_counts, merge_seq, construct_local_pair_counts
import regex as re
from collections import Counter
import multiprocessing as mp
import time
import json
from tests.common import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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