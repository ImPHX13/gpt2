import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # Number of tokens per shard

# Create the local directory for caching if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['']  # End of text token

def download_dataset():
    """Download the dataset from Hugging Face."""
    return load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

def tokenize(doc):
    """
    Tokenize a single document.
    
    Args:
        doc (dict): A document containing text to be tokenized.
        
    Returns:
        np.ndarray: Tokenized document as an array of uint16 tokens.
    """
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

def write_datafile(filename, tokens_np):
    """
    Save tokens to a .npy file.
    
    Args:
        filename (str): The path to the output file.
        tokens_np (np.ndarray): Array of uint16 tokens to be saved.
    """
    np.save(filename, tokens_np)

def process_dataset(fw):
    """
    Tokenize the dataset and save the tokens in shards.
    
    Args:
        fw (Dataset): The dataset to be processed.
    """
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                # Append tokens to the current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # Write the current shard to file and start a new shard
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                
                shard_index += 1
                progress_bar = None
                
                # Start a new shard with the leftover tokens
                all_tokens_np[:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder
        
        if token_count != 0:
            # Write any remaining tokens as the last shard
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

if __name__ == "__main__":
    dataset = download_dataset()
    process_dataset(dataset)
