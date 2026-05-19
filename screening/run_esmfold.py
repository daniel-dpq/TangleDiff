import os
import logging
import subprocess
import sys
from Bio.PDB import PDBParser
from typing import List, Tuple
import torch
import esm
from omegaconf import DictConfig
import typing as T
from tqdm import tqdm
import argparse
from Bio import SeqIO
import time



def create_batched_sequence_datasest(
    sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:

    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences


def run_esmfold(seqs: T.List[T.Tuple[str, str]], pdb_dir: str, device: str = "cuda", chunk_size: int = 64, max_tokens_per_batch: int = 320):
    '''
    Monomer sequence prediction only
    '''
    seqs = sorted(seqs, key=lambda x: len(x[1]))
    seqs_to_pred = [(header, seq.replace('/', ':')) for header, seq in seqs if not os.path.exists(os.path.join(pdb_dir, f"{header}.pdb"))]
    logging.info(f"Predicting the structure for {len(seqs_to_pred)} sequences")

    if len(seqs_to_pred) > 0:
        model = esm.pretrained.esmfold_v1()
        model = model.eval().to(device)
        model.set_chunk_size(chunk_size)        
        batched_sequences = list(create_batched_sequence_datasest(seqs_to_pred, max_tokens_per_batch=max_tokens_per_batch))
        for headers, sequences in tqdm(batched_sequences, desc="Predicting structures with ESMFold"):
            output = model.infer(sequences)
            output = {key: value.cpu() for key, value in output.items()}
            pdbs = model.output_to_pdb(output)
            for header, pdb_string in zip(headers, pdbs):
                path = os.path.join(pdb_dir, f"{header}.pdb")
                with open(path, "w") as f:
                    f.write(pdb_string)
        del model
        torch.cuda.empty_cache()  



def main():
    parser = argparse.ArgumentParser(description="Run ESMFold on a list of sequences")
    parser.add_argument("--fasta_file", type=str, required=True, help="Path to the input file containing sequences in FASTA format")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory to save the predicted PDB files")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda)")
    parser.add_argument("--range", type=str, default=None, help="Range of sequences to process (e.g., '0-100')")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size for processing sequences (default: 64)")
    parser.add_argument("--max_tokens_per_batch", type=int, default=8172, help="Maximum tokens per batch (default: 320)")
    parser.add_argument("--split", type=str, default=None, help="Split the sequences into chunks of this size (default: 16)")
    args = parser.parse_args()

    os.makedirs(args.pdb_dir, exist_ok=True)

    sequences = [(record.id, str(record.seq)) for record in SeqIO.parse(args.fasta_file, "fasta")]
    sequences = [(header, seq) for header, seq in sequences if not os.path.exists(os.path.join(args.pdb_dir, f"{header}.pdb"))]
    if len(sequences) == 0:
        return
    
    if args.range:
        start, end = map(int, args.range.split('-'))
        sequences = sequences[16*start:16*end]
    if args.split:
        sequences = [(header, seq) for header, seq in sequences if not os.path.exists(os.path.join(args.pdb_dir, f"{header}.pdb"))]
        start, total = map(int, args.split.split('/'))
        num_per_chunk = int(len(sequences) / total) + 1
        sequences = sequences[start*num_per_chunk:(start+1)*num_per_chunk]
        time.sleep(30)

    run_esmfold(sequences, args.pdb_dir, device=args.device, chunk_size=args.chunk_size, max_tokens_per_batch=args.max_tokens_per_batch)


if __name__ == "__main__":
    main()