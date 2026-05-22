import os
import torch
import esm
from Bio import SeqIO, Align
from Bio.PDB import PDBParser
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import typing as T
import logging
from transformers import EsmTokenizer, EsmForMaskedLM
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler  # used to define color cycles
import argparse


def get_aa_freq(seqs):
    aa_freq = {}
    for seq in seqs:
        for aa in seq:
            if aa == 'X':
                continue
            if aa in aa_freq:
                aa_freq[aa] += 1
            else:
                aa_freq[aa] = 1
    return aa_freq

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


def process_sequence(seq):
    try:
        start_idx = seq.index('<cls>')
        end_idx = min(seq.index('<eos>'), len(seq)-1)
    except:
        return ''
    if start_idx >= end_idx:
        return ''
    return seq[start_idx+5:end_idx]


def is_protein(seq):
    if len(seq) == 0:
        return False
    return all([aa in 'ACDEFGHIKLMNPQRSTVWYX' for aa in seq])


def esm_embed(seqs: List[Tuple[str, str]]):

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()
    batched_sequences = create_batched_sequence_datasest(seqs, max_tokens_per_batch=512*100)
    batched_sequences = list(batched_sequences)

    seq_repr = []
    for headers, sequences in tqdm(batched_sequences, desc="ESM Embedding"):
        _, _, batch_tokens = batch_converter(list(zip(headers, sequences)))
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)["representations"][33].cpu().numpy()
        for i, tokens_len in enumerate(batch_lens):
            seq_repr.append(results[i, 1 : tokens_len - 1].mean(0))
    
    return np.array(seq_repr)


def get_mean_plddt(pdb_path):
    # use biopython to get b factor
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('X', pdb_path)
    model = structure[0]
    plddt = []
    for chain in model:
        for residue in chain:
            for atom in residue:
                plddt.append(atom.get_bfactor())
    return np.mean(plddt)


def plddt(seqs: List[Tuple[str, str]], pdb_dir: str, fasta_name: str):

    seqs = sorted(seqs, key=lambda x: len(x[1]))
    seqs_to_pred = [(header, seq+":"+seq) for header, seq in seqs if not os.path.exists(os.path.join(pdb_dir, f"{fasta_name}_{header}.pdb"))]
    logging.info(f"Predicting the structure for {len(seqs_to_pred)} sequences")

    if len(seqs_to_pred) > 0:
        model = esm.pretrained.esmfold_v1()
        model = model.eval().cuda()
        model.set_chunk_size(256)        
        batched_sequences = list(create_batched_sequence_datasest(seqs_to_pred, max_tokens_per_batch=5120))
        for headers, sequences in tqdm(batched_sequences, desc="Predicting the structure"):
            output = model.infer(sequences)
            output = {key: value.cpu() for key, value in output.items()}
            pdbs = model.output_to_pdb(output)
            for header, pdb_string in zip(headers, pdbs):
                path = os.path.join(pdb_dir, f"{fasta_name}_{header}.pdb")
                with open(path, "w") as f:
                    f.write(pdb_string)
        del model
        torch.cuda.empty_cache()    

    pdb_paths = [os.path.join(pdb_dir, f"{fasta_name}_{header}.pdb") for header, _ in seqs]
    plddts = [get_mean_plddt(pdb_path) for pdb_path in tqdm(pdb_paths, desc="Calculating the plddt")]
    
    return np.array(plddts)


def parse_blast_result(processed_fasta, ids, train_db):
    logging.info(f"Calculating the sequence similarity...")
    cmd = f'blastp -query {processed_fasta} -db {train_db} -out {blast_file} -evalue 0.00001 -outfmt 6 -num_alignments 1'
    if not os.path.exists(blast_file):
        os.system(cmd)

    df = pd.read_csv(blast_file, sep='\t', header=None)
    df.columns = ['query', 'subject', 'identity', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
    identities = []
    for id_ in tqdm(ids, desc="Parsing the blast result"):
        identity = df[df['query'] == id_]['identity'].values
        if len(identity) > 0:
            identities.append(identity[0])
        else:
            identities.append(0)
    return identities


def esm_pseudo_perplexity(seqs: List[str]):

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to('cuda').eval()

    ppls = []
    seqs = tqdm(seqs, desc="Calculating the pseudo perplexity")
    for seq in seqs:
        tensor_input = tokenizer.encode(seq, return_tensors='pt')   # 1, L
        repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)  # L-2, L
        # mask one by one except [CLS] and [SEP]
        mask = torch.ones(tensor_input.size(-1) -1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)

        if masked_input.size(0) <= 260:
            with torch.no_grad():
                loss = model(masked_input.to('cuda'), labels=labels.to('cuda')).loss.cpu()
        else:
            # calculate for two times to avoid memory error
            masked_input_1 = masked_input[:260]
            labels_1 = labels[:260]
            masked_input_2 = masked_input[260:]
            labels_2 = labels[260:]
            with torch.no_grad():
                loss_1 = model(masked_input_1.to('cuda'), labels=labels_1.to('cuda')).loss.cpu()
                loss_2 = model(masked_input_2.to('cuda'), labels=labels_2.to('cuda')).loss.cpu()
                loss = (loss_1 * 260 + loss_2 * (masked_input.size(0) - 260)) / masked_input.size(0)
        
        ppl = np.exp(loss.item())
        ppls.append(ppl)
        torch.cuda.empty_cache()
    
    return np.array(ppls)


def calculate(fasta_file, train_db):
    logging.basicConfig(level=logging.INFO)
    fasta_file = fasta_file
    fasta_dir = os.path.dirname(fasta_file)
    fasta_name = os.path.basename(fasta_file)[:-6]
    pdb_dir = os.path.join(fasta_dir, 'esmfold_pdb')
    result_dir = os.path.join(fasta_dir, 'result')
    blast_file = os.path.join(result_dir, f'proteins_blastp_{fasta_name}.tsv')
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    records = list(SeqIO.parse(fasta_file, "fasta"))
    seqs = [str(seq.seq) for seq in records]
    ids = [seq.id for seq in records]

    seqs_with_id = [(id_, seq) for id_, seq in zip(ids, seqs) if is_protein(seq)]
    ids = [id_ for id_, _ in seqs_with_id]
    seqs = [seq for _, seq in seqs_with_id]

    similarities = parse_blast_result(fasta_file, ids, blast_file, train_db)
    plddts = plddt(seqs_with_id, pdb_dir, fasta_name)
    pdb_files = [os.path.join(pdb_dir, f'{fasta_name}_{id_}.pdb') for id_ in ids]
    ppls = esm_pseudo_perplexity(seqs)
    lengths = [len(seq) for seq in seqs]
    
    df = pd.DataFrame({
        'id': ids,
        'length': lengths,
        'ppl': ppls,
        'plddt': plddts,
        'similarity': similarities,
    })

    df.to_csv(os.path.join(result_dir, f'result_{fasta_name}.csv'), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_file', type=str, required=True)
    parser.add_argument('--train_db', type=str, required=True)
    args = parser.parse_args()
    calculate(args.fasta_file, args.train_db)
