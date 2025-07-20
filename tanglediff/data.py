from torch.utils.data import Dataset
from typing import List, Sequence
import torch
import logging
import esm
import pandas as pd
import numpy as np
import random
from tape import TAPETokenizer, ProteinBertForMaskedLM, UniRepModel

from omegaconf import ListConfig

import tanglediff.residue_constant as rc


def get_length(string):
    if '<mask>' not in string and '<pad>' not in string:
        return len(string)
    l = 0   # counter
    p = 0   # pointer
    while p < len(string):
        if string[p] in rc.aa_1:
            l += 1
            p += 1
        else:
            if string[p:p+5] == '<pad>':
                l += 1
                p += 5
            else:
                assert string[p:p+6] == '<mask>', f"string[p:p+6]={string[p:p+6]}"
                l += 1
                p += 6
    return l


def pstring_to_list(string:str) -> List[str]:
    """
    convert a string with <mask> to a list of aa
    """
    if '<mask>' not in string:
        return list(string)
    l = []
    p = 0   # pointer
    while p < len(string):
        if string[p] in rc.aa_1:
            l.append(string[p])
            p += 1
        else:
            if string[p:p+5] == '<pad>':
                l.append('<pad>')
                p += 5
            else:
                assert string[p:p+6] == '<mask>', f"string[p:p+6]={string[p:p+6]}"
                l.append('<mask>')
                p += 6
    return l



class SequenceMapper:
    def __init__(
        self, 
        max_length: int, 
        seq_encode_mode: str, 
        logits=True,  # only for esm, use latent embeddings if False, logits if True
        device='cuda'
    ):
        self.max_length = max_length
        self.seq_encode_mode = seq_encode_mode
        self.logits = logits
        self.device = device
        self.standardizer = lambda x: x
        self.rev_standardizer = lambda x: x

        if 'esm' in self.seq_encode_mode:
            if self.seq_encode_mode == 'esm2_t33_650M_UR50D':
                if self.logits:
                    self.standardizer = lambda x: (x + 0.7) / 2
                    self.rev_standardizer = lambda x: x * 2 - 0.7
                else:
                    self.standardizer = lambda x: x / 0.278
                    self.rev_standardizer = lambda x: x * 0.278
                self.channel_dim = 1280
                self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            elif self.seq_encode_mode == 'esm2_t36_3B_UR50D':
                if self.logits:
                    self.standardizer = lambda x: (x + 1) / 2.6
                    self.rev_standardizer = lambda x: x * 2.6 - 1
                self.channel_dim = 2560
                self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            elif self.seq_encode_mode == 'esm2_t30_150M_UR50D':
                if self.logits:
                    self.standardizer = lambda x: (x + 0.6) / 2
                    self.rev_standardizer = lambda x: x * 2 - 0.6
                self.channel_dim = 640
                self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            else:
                logging.error(f"seq_encode_mode={self.seq_encode_mode} is not supported")
            self.esm_model.requires_grad_(False)
            self.esm_model.to(self.device)
            if self.logits:
                self.channel_dim = 20   # 20 standard amino acids
                self.standard_toks = self.esm_alphabet.standard_toks[:20]
                self.standard_esm_idx = [self.esm_alphabet.get_idx(tok) for tok in self.standard_toks]
                self.tok_to_idx = {tok: i for i, tok in enumerate(self.standard_toks)}
            else:
                self.standard_toks = self.esm_alphabet.all_toks

        elif self.seq_encode_mode == "VHSE":
            self.channel_dim = len(rc.VHSE_scales["A"])

        elif self.seq_encode_mode == "tape_base":
            self.standardizer = lambda x: (x + 2.3) / 6.4
            self.rev_standardizer = lambda x: x * 6.4 - 2.3
            self.model = ProteinBertForMaskedLM.from_pretrained('bert-base')
            self.model.requires_grad_(False)
            self.model.to(self.device)
            if self.logits:
                self.tokenizer = TAPETokenizer(vocab='iupac')
                self.channel_dim = 20
                self.standard_toks = list('ACDEFGHIKLMNPQRSTVWY')
                self.standard_bert_idx = self.tokenizer.convert_tokens_to_ids(self.standard_toks)
                self.tok_to_idx = {tok: i for i, tok in enumerate(self.standard_toks)}
            else: 
                raise NotImplementedError(f"tape_base with embeddings is not supported")
        
        elif self.seq_encode_mode == "tape_unirep":
            self.standardizer = lambda x: (x + 0) / 0.2
            self.rev_standardizer = lambda x: x * 0.2 - 0
            self.model = UniRepModel.from_pretrained('babbler-1900')
            self.model.requires_grad_(False)
            self.model.to(self.device)
            if self.logits:
                self.tokenizer = TAPETokenizer(vocab='unirep')
                self.channel_dim = 20
                self.standard_toks = list('ACDEFGHIKLMNPQRSTVWY')
                self.standard_bert_idx = self.tokenizer.convert_tokens_to_ids(self.standard_toks)
                self.tok_to_idx = {tok: i for i, tok in enumerate(self.standard_toks)}
            else: 
                raise NotImplementedError(f"tape_base with embeddings is not supported")
    
    def encode(self, batch_seq: List[str])->List[torch.Tensor]:
        """
        batch_seq: a list of sequences
        return 
                batch_embed: (B, L, C)
                batch_mask: (B, L), 1 for valid aa, 0 for padding
        """
        batch_seq = list(map(pstring_to_list, batch_seq))
        if self.max_length is None:
            max_length = max(map(len, batch_seq))
        else:
            max_length = self.max_length

        batch_mask = torch.tensor(
            [
                [1] * len(seq) + [0] * (max_length - len(seq)) 
                for seq in batch_seq
            ], 
        )

        if self.seq_encode_mode == "VHSE":
            batch_embed = list(map(lambda seq: [rc.VHSE_scales[aa] for aa in seq], batch_seq))
            pad_token = [0.0] * len(rc.VHSE_scales["A"])
            for i in range(len(batch_embed)):
                batch_embed[i].extend([pad_token] * (max_length - len(batch_embed[i])))
            batch_embed = torch.tensor(batch_embed) # [B, MAX_L, 8]

        elif 'esm' in self.seq_encode_mode:
            esm_batch_converter = self.esm_alphabet.get_batch_converter()
            # if '<mask>' not in batch_seq[0]: # normal encoding
            batch_seq = [('a', "".join(seq)) for seq in batch_seq] # add dummy labels for esm list[seq] -> list[(label, seq)]
            _, _, batch_tokens = esm_batch_converter(batch_seq) # [B, L+2] cls and eos tokens added
            # padd to max_length
            pad_length = max_length + 2 - batch_tokens.shape[-1]
            pad_tensor = torch.zeros(
                batch_tokens.shape[0], 
                pad_length, 
                dtype=batch_tokens.dtype
            ) + self.esm_alphabet.padding_idx
            batch_tokens = torch.cat([batch_tokens, pad_tensor], dim=1).to(self.device) # [B, MAX_L+2]
            self.esm_model.eval()
            # get logits or embeddings
            with torch.no_grad():
                if self.logits:
                    batch_embed = self.esm_model(                
                        batch_tokens,
                        repr_layers=[],
                        return_contacts=False,
                    )["logits"].cpu() # [B, MAX_L+2, TOKEN_NUM]
                    batch_embed = batch_embed[..., 1:-1, self.standard_esm_idx[0]:self.standard_esm_idx[-1]+1]    # [B, MAX_L, 20]
                else:
                    batch_embed = self.esm_model(                
                        batch_tokens,
                        repr_layers=[self.esm_model.num_layers],
                        return_contacts=False,
                    )["representations"][self.esm_model.num_layers].cpu()
                    batch_embed = batch_embed[..., 1:-1, :]    # [B, MAX_L, EMB_DIM]

        elif self.seq_encode_mode == "tape_base" or self.seq_encode_mode == "tape_unirep":
            batch_seq = ["".join(seq) for seq in batch_seq]
            tokens = [self.tokenizer.encode(seq) for seq in batch_seq]
            batch_tokens = torch.full(
                (len(tokens), max_length+2), 
                self.tokenizer.convert_token_to_id('<pad>'), 
                device=self.device
            )   # [B, MAX_L+2]
            for i, token in enumerate(tokens):
                batch_tokens[i, :len(token)] = torch.tensor(token, device=self.device)
            self.model.eval()
            with torch.no_grad():
                assert self.logits
                batch_embed = self.model(batch_tokens)[0]   # [B, MAX_L+2, TOKEN_NUM]
                batch_embed = batch_embed[..., 1:-1, self.standard_bert_idx]   # [B, MAX_L, 20]

        else:
            raise NotImplementedError(f"seq_encode_mode={self.seq_encode_mode}")

        batch_embed = self.standardizer(batch_embed)

        return batch_embed, batch_mask

    def decode(self, batch_embed: torch.Tensor, batch_mask: torch.Tensor=None)->List[str]:
        """
        batch_embed: B, L, C
        batch_mask: B, L
        """
        if batch_mask is None:
            batch_mask = torch.ones(batch_embed.shape[0], batch_embed.shape[1])

        batch_embed = self.rev_standardizer(batch_embed)

        if self.seq_encode_mode == "VHSE":
            # [aa_num, 8] -> [1, 1, aa_num, 8]
            VHSE_values = torch.tensor(rc.VHSE_scales_values, device=batch_embed.device).unsqueeze(0).unsqueeze(0)
            # [B, MAX_L, 8] -> [B, MAX_L, 1, 8]
            batch_embed = batch_embed.unsqueeze(-2)
            # B, MAX_L, aa_num, 8 -> B, MAX_L, aa_num
            dist = torch.sqrt(torch.sum(torch.square(VHSE_values - batch_embed), dim=-1) / 8.0)
            # B, MAX_L, aa_num -> B, MAX_L
            index = torch.argmin(dist, dim=-1)

            batch_seq = []
            for seq_index, mask in zip(index, batch_mask):
                seq = ''
                for i, m in zip(seq_index, mask):
                    if m == 0:
                        break
                    seq += rc.VHSE_scales_aa[i]
                batch_seq.append(seq)
        
        elif 'esm' in self.seq_encode_mode:
            if self.logits:
                # logits to tokens [B, MAX_L, TOKEN_NUM] -> [B, MAX_L]
                batch_tokens = batch_embed.argmax(dim=-1)
            else:
                with torch.no_grad():
                    # embeddings to logits [B, MAX_L, EMB_DIM] -> [B, MAX_L, TOKEN_NUM]
                    logits = self.esm_model.lm_head(batch_embed.to(self.device)).cpu()
                    # logits to tokens [B, MAX_L, TOKEN_NUM] -> [B, MAX_L]
                    batch_tokens = logits.argmax(dim=-1)
            # tokens to seqs
            batch_seq = []
            for tokens, mask in zip(batch_tokens, batch_mask):
                seq = ''
                for token, m in zip(tokens, mask):
                    if m == 0:
                        break
                    seq += self.standard_toks[token]
                batch_seq.append(seq)
        
        elif self.seq_encode_mode == "tape_base" or self.seq_encode_mode == "tape_unirep":
            assert self.logits
            # logits to tokens [B, MAX_L, 20] -> [B, MAX_L]
            batch_tokens = batch_embed.argmax(dim=-1)
            # tokens to seqs
            batch_seq = []
            for tokens, mask in zip(batch_tokens, batch_mask):
                seq = ''
                for token, m in zip(tokens, mask):
                    if m == 0:
                        break
                    seq += self.standard_toks[token]
                batch_seq.append(seq)

        else:
            raise NotImplementedError(f"seq_decode_mode={self.seq_encode_mode}")

        return batch_seq

    def test(self):
        batch_seq = ["MFGHIKLMNPQRSTVWY", "MKLMNPQRSTVW"]
        batch_embed, batch_mask = self.encode(batch_seq)
        batch_seq_decoded = self.decode(batch_embed, batch_mask)
        assert batch_seq == batch_seq_decoded, f"batch_seq={batch_seq}, batch_seq_decoded={batch_seq_decoded}"
        print(f"seq_encode_mode={self.seq_encode_mode} test passed")


class SequenceDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_length = self.cfg.max_length
        self.be_seps = self.cfg.binding_energy_seps
        self.be_num_classes = len(self.be_seps) + 1
        random.seed(self.cfg.seed)
        
        # read data files
        df = pd.read_csv(self.cfg.data_file)
        df = df[df['sequence'].apply(len) == df['modeled_seq_len']] # remove sequences with wrong length
        df = df.dropna()
        df = df[df['symmetry'] == 'c2']
        df = df[df['sym_rmsd'] <= self.cfg.max_sym_rmsd]
        df = df[df['core_binding_energy_relax_pack'] < self.cfg.max_binding_energy]
        
        if self.cfg.use_core:
            df = df[df['core_BSA'] >= self.cfg.min_bsa]
            df = df[df['core_gln'].abs() >= self.cfg.min_gln]
            df = df[df['core_gln'].abs() <= 3]
            df = df[df['core_plddt'] >= self.cfg.min_plddt]
            if self.max_length:
                df = df[df['core_modeled_seq_len'] <= self.max_length] 
            self.seq_ids = df['acc_num'].values
            self.sequences = df['core_sequence'].values
            self.lengths = df['core_modeled_seq_len'].astype('int32').values
        else:
            df = df[df['bsa'] >= self.cfg.min_bsa]
            df = df[df['gln'].abs() >= self.cfg.min_gln]
            df = df[df['gln'].abs() <= 3]
            df = df[df['plddt'] >= self.cfg.min_plddt]
            if self.max_length:
                df = df[df['modeled_seq_len'] <= self.max_length]
            self.seq_ids = df['acc_num'].values
            self.sequences = df['sequence'].values
            self.lengths = df['modeled_seq_len'].values
        df.reset_index(drop=True, inplace=True)
        
        self.be_classes = df['core_binding_energy_relax_pack'].apply(self._discrete_binding_energy).values
        
        if self.max_length is None:
            self.max_length = max(self.lengths)

        self.clusters = df['cluster50'].sort_values().unique().tolist()
        self.clusters = [list(df[df['cluster50'] == i].index) for i in self.clusters]
        # debug
        clusters = [i for cluster in self.clusters for i in cluster]
        assert len(clusters) == len(self.seq_ids), f"len(clusters)={len(clusters)}, len(self.seq_ids)={len(self.seq_ids)}"
    
    def _discrete_binding_energy(self, binding_energy):
        """
        return discrete binding energy
        """
        for i, sep in enumerate(self.be_seps):
            if binding_energy < sep:
                return i
        return len(self.be_seps)

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, cluster_idx):
        idxes = self.clusters[cluster_idx]
        idx = random.choice(idxes)
        return self.seq_ids[idx], self.sequences[idx], self.be_classes[idx]
    
    def sample_lengths(self, batch_size, be_class=None, length_restrict=None)->torch.Tensor:
        """
        sample lengths for sampling
        return mask with shape (batch_size, max_length), 1 for valid aa, 0 for padding
        """
        if be_class is None or be_class == self.be_num_classes:
            lengths = self.lengths
        else:
            indices = [i for i, x in enumerate(self.be_classes) if x == be_class]
            lengths = np.array(self.lengths)[indices]
        if length_restrict is not None:
            if type(length_restrict) == int:
                lengths = [length_restrict]
            elif type(length_restrict) == ListConfig:
                assert len(length_restrict) == 2
                lengths = [l for l in lengths if l >= length_restrict[0] and l <= length_restrict[1]]
            else:
                raise NotImplementedError(f"length_restrict={length_restrict}")
        assert len(lengths) > 0, f"Invalid length specified: {length_restrict}"
        max_length = max(lengths)
        lengths = np.random.choice(lengths, batch_size)
        mask = torch.zeros((batch_size, max_length), device='cpu')
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        return mask


class SequenceBatchCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        batch_id, batch_seq, batch_be_class = zip(*batch)
        max_length = max(map(len, batch_seq))
        batch_be_class = torch.tensor(batch_be_class, device='cpu')

        batch = {
            "id": batch_id,
            "seqs": batch_seq,
            "be_class": batch_be_class,
        }

        return batch
    
    def _pad_with_zero(self, ragged_list, max_length)->torch.Tensor:
        """
        pad tensor with zeros
        Args:
            ragged_list: a list of tensors with shapes (L_i, ...)
            max_length: max length
        """
        ragged_list = list(map(lambda x: torch.tensor(x, device='cpu'), ragged_list))
        tensor = torch.zeros((len(ragged_list), max_length, *ragged_list[0].shape[1:]), device='cpu')
        for i, t in enumerate(ragged_list):
            tensor[i, :t.shape[0]] = t
        return tensor
        
    

if __name__ == "__main__":
    mapper = SequenceMapper(max_length=None, seq_encode_mode="esm2_t12_35M_UR50D", device="cpu")
    mapper.test()
    mapper = SequenceMapper(max_length=500, seq_encode_mode="VHSE", device="cpu")
    mapper.test()
    # mapper = SequenceMapper(max_length=50, seq_encode_mode="esm2_t12_35M_UR50D", device="cuda", logits=False)
    # mapper.test()
    # mapper = SequenceMapper(max_length=50, seq_encode_mode="VHSE", device="cuda")
    # mapper.test()