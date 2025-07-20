# from tanglediffusion.ss import pdb_to_ss_string, loop_pos_from_ss_string, bp_from_ss_string
import numpy as np
import Bio.PDB
from Bio.PDB import is_aa
import warnings
import os

def backb_pos_to_gln_matrix(ca_pos_0: np.ndarray, ca_pos_1: np.ndarray,)-> np.ndarray:
    '''
        Calculate GLN matrix
        Args:
            ca_pos0: [*, N1, 3], [CA]
            ca_pos1: [*, N2, 3], [CA]
        Return:
            GLN matrix: [*, N1-1, N2-1]
    '''

    n1 = ca_pos_0.shape[-2]
    n2 = ca_pos_1.shape[-2]

    # [*, N1-1, 3], [*, N2-1, 3]
    r_0 = (ca_pos_0[..., 1:, :] + ca_pos_0[..., :-1, :]) / 2
    r_1 = (ca_pos_1[..., 1:, :] + ca_pos_1[..., :-1, :]) / 2
    delta_r_0 = ca_pos_0[..., 1:, :] - ca_pos_0[..., :-1, :]
    delta_r_1 = ca_pos_1[..., 1:, :] - ca_pos_1[..., :-1, :]

    # [*, N1-1, N2-1, 3]
    r_0 = np.repeat(r_0[..., np.newaxis, :], n2-1, axis=-2)
    r_1 = np.repeat(r_1[..., np.newaxis, :, :], n1-1, axis=-3)
    delta_r_0 = np.repeat(delta_r_0[..., np.newaxis, :], n2-1, axis=-2)
    delta_r_1 = np.repeat(delta_r_1[..., np.newaxis, :, :], n1-1, axis=-3)

    # [*, N1-1, N2-1]
    numerator = np.matmul(
        (r_0 - r_1)[..., np.newaxis, :], 
        np.cross(delta_r_0, delta_r_1, axis=-1)[..., np.newaxis]
    ).squeeze(axis=-1).squeeze(axis=-1)
    denomnator = np.power(np.linalg.norm(r_0 - r_1, axis=-1), 3) * 4 * np.pi

    assert(numerator.shape[-1] == n2-1 and numerator.shape[-2] == n1-1)

    gln_matrix = numerator / denomnator

    # replace nan with 0
    gln_matrix = np.nan_to_num(gln_matrix, nan=0.0)

    return gln_matrix


def backb_pos_from_pdb(pdb_path) -> np.ndarray:
    '''
        Extract CA positions from PDB file
        Args:
            pdb_path: str, path to PDB file
        Return:
            ca_pos: A list of numpy array, for the ith chain: [Ni, 3]
    '''
    if pdb_path.endswith('.cif'):
        parser = Bio.PDB.MMCIFParser(QUIET=True)
        structure = parser.get_structure('X', pdb_path)
        ca_pos = []
        model = structure[0]
        chain_ids = [chain.id for chain in model]
        chain = model[chain_ids[0]]
        for residue in chain:
            if is_aa(residue):
                alpha_carbon = residue['CA']
                ca_pos.append(alpha_carbon.get_coord())
        ca_pos = np.array([ca_pos])
        return ca_pos
    
    parser = Bio.PDB.PDBParser(QUIET=True)

    structure = parser.get_structure('X', pdb_path)
    ca_pos = []
    for model in structure:
        for chain in model:
            chain_ca_pos = []
            for residue in chain:
                # if has no CA atom, skip
                if 'CA' in residue:
                    chain_ca_pos.append(residue['CA'].get_coord())
            ca_pos.append(np.array(chain_ca_pos))
    
    return ca_pos


def pdb_to_gln_matrix(pdb_path: str, num_chain: int) -> np.ndarray:
    '''
        Calculate GLN matrix from PDB file
        Args:
            pdb_path: str, path to PDB file
            gln_path: str, path to save GLN matrix
    '''
    if num_chain != 1 and num_chain != 2:
        raise ValueError('Number of chains should be 1 or 2')
    
    ca_pos = backb_pos_from_pdb(pdb_path)

    if len(ca_pos) != num_chain:
        raise ValueError(f'Number of chains does not match: {pdb_path}')
    
    if num_chain == 1:
        warnings.filterwarnings("ignore")
        gln_matrix = backb_pos_to_gln_matrix(ca_pos[0], ca_pos[0])
        warnings.resetwarnings()
    else:
        gln_matrix = backb_pos_to_gln_matrix(ca_pos[0], ca_pos[1])

    return gln_matrix


# def pdb_to_core_gln_matrix(pdb_path: str, num_chain: int) -> np.ndarray:
#     if num_chain != 1 and num_chain != 2:
#         raise ValueError('Number of chains should be 1 or 2')
    
#     ss_string = pdb_to_ss_string(pdb_path)
#     if ss_string is None:
#         return None
    
#     if num_chain == 2:
#         ss_string = ss_string[:len(ss_string)//2]
#     # find the index of the first H/E and last H/E
#     H_start = ss_string.find('H')
#     E_start = ss_string.find('E')
#     if H_start == -1 and E_start == -1:
#         start = 0
#         end = 0
#     elif H_start == -1:
#         start = E_start
#         end = ss_string.rfind('E')
#     elif E_start == -1:
#         start = H_start
#         end = ss_string.rfind('H')
#     else:
#         start = min(H_start, E_start)
#         end = max(ss_string.rfind('H'), ss_string.rfind('E'))
#     if start >= end:
#         return None


#     ca_pos = backb_pos_from_pdb(pdb_path)

#     if len(ca_pos) != num_chain:
#         raise ValueError(f'Number of chains does not match: {pdb_path}')
    
#     if num_chain == 1:
#         warnings.filterwarnings("ignore")
#         gln_matrix = backb_pos_to_gln_matrix(ca_pos[0][start:end], ca_pos[0][start:end])
#         warnings.resetwarnings()
#     else:
#         gln_matrix = backb_pos_to_gln_matrix(ca_pos[0][start:end], ca_pos[1][start:end])

#     return gln_matrix
