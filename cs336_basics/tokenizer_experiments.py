from bpe_utils import Tokenizer
import regex as re
import random
from pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
import argparse
import time

def sample_documents(file_path, n_doc, num_processes=4):
    DOC_SPLIT_PAT = re.escape("<|endoftext|>")
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [
        (file_path, start, end, DOC_SPLIT_PAT, n_doc) 
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    # sample n_doc per chunk
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=num_processes) as pool:
        results = pool.map(sample_documents_per_chunk, tasks)
    
    results_combined = []
    for result in results:
        results_combined.extend(result)
    
    # sample n_doc after combined
    final_samples = random.choices(results_combined, k=n_doc)
    return final_samples


def sample_documents_per_chunk(args):
    input_path, start, end, doc_split_pat, n_doc = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    docs = re.split(doc_split_pat, chunk)
    samples = random.choices(docs, k=n_doc)
    return samples

def parse_args():
    p = argparse.ArgumentParser(description="Tokenizer experiments")
    p.add_argument(
        "--text-file-path",
        type=str,
        required=True,
        help="Path to corpus text file",
    )
    p.add_argument(
        "--vocab-file-path",
        type=str,
        required=True,
        help="Path to vocab file",
    )
    p.add_argument(
        "--merges-file-path",
        type=str,
        required=True,
        help="Path to merges file",
    )
    p.add_argument(
        "--special-tokens",
        type=str,
        default="<|endoftext|>",
        help="Comma-separated special tokens",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of random docs to print",
    )
    p.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of processes/chunks to run sample doc from",
    )
    return p.parse_args()

def main():
    args = parse_args()
    special_tokens = sorted(args.special_tokens.split(","), key=len, reverse=True)

    tokenizer = Tokenizer.from_files(args.vocab_file_path, args.merges_file_path, special_tokens=special_tokens)
    sampled_docs = sample_documents(args.text_file_path, args.num_samples, num_processes=args.num_processes)
    n_bytes, n_tokens = 0, 0
    start = time.perf_counter()
    for doc in sampled_docs:
        n_tokens += len(tokenizer.encode(doc))
        n_bytes += len(doc.encode("utf-8"))
    end = time.perf_counter()

    print(f"compression ratio: {n_bytes/n_tokens:.4f} bytes/token")
    print(f"average throughput: {n_bytes/(end - start):.4f} bytes/second")

if __name__ == "__main__":
    main()

# uv run tokenizer_experiments.py --text-file-path ../data/TinyStoriesV2-GPT4-train.txt --vocab-file-path vocab_tinystories.json --merges-file-path merges_tinystories.txt 
# compression ratio: 4.1607 bytes/token
# average throughput: 19955.2924 bytes/second

# uv run tokenizer_experiments.py --text-file-path ../data/owt_train.txt --vocab-file-path vocab_owt.json --merges-file-path merges_owt.txt --num-processes 12
# compression ratio: 4.4899 bytes/token
# average throughput: 4845.2011 bytes/second

# uv run tokenizer_experiments.py --text-file-path ../data/owt_train.txt --vocab-file-path vocab_tinystories.json --merges-file-path merges_tinystories.txt --num-processes 12
# compression ratio is 2.2272237043764544
# compression ratio is 3.27283451297607
# compression ratio is 3.4338842975206614
# on average every token represents fewer bytes, which is expected.