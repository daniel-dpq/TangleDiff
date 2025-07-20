from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import PDBxFile, get_structure, get_model_count
from biotite.application import dssp

from itertools import groupby
from collections import Counter
from typing import Tuple
import os
import mdtraj as md

import numpy as np


def is_helix(ss):
    return ss == "H" or ss == "G" or ss == "I"


def is_sheet(ss):
    return ss == "E"


def ss_to_index(ss):
    """Secondary structure symbol to index.
    Helix=1
    Sheet=2
    Coil=0
    """
    if is_helix(ss):
        return 1
    if is_sheet(ss):
        return 2
    else:
        return 0


def pdb_to_ss_string(fname: str) -> str:
    """Count the secondary structures (# alpha, # beta) in the given pdb file"""
    assert os.path.exists(fname)

    try:
        traj = md.load(fname)
        pdb_ss = md.compute_dssp(traj, simplified=True)
        ss_string = ''.join(pdb_ss[0])
    except:
        ss_string = None
    
    return ss_string



def loop_pos_from_ss_string(ss_string) -> np.ndarray:
    """Get the loop positions from the secondary structure string.
    A loop is defined as a sequence of coil residues.
    """
    loop_pos = []
    loop_start = None
    for i, ss in enumerate(ss_string):
        if is_helix(ss) or is_sheet(ss):
            if loop_start is not None:
                loop_pos.append((loop_start, i))
                loop_start = None
        else:
            if loop_start is None:
                loop_start = i
    if loop_start is not None:
        loop_pos.append((loop_start, len(ss_string)))
        
    return np.array(loop_pos)


def bp_from_ss_string(ss_string) -> np.ndarray:
    """Get the break points according to loop positions.
    """
    bp = []
    loop_start = None
    for i, ss in enumerate(ss_string):
        if is_helix(ss) or is_sheet(ss):
            if loop_start is not None:
                bp.append(loop_start)
                bp.append(i-1)
                loop_start = None
        else:
            if loop_start is None:
                loop_start = i
    if loop_start is not None:
        bp.append(loop_start)
        bp.append(len(ss_string)-1)
        
    return np.array(bp)


if __name__ == "__main__":
    pdb_path = '/home/puqing/design/2b3p.pdb'
    pdb_path = ''

    ss_string = pdb_to_ss_string_v2(pdb_path)
    print(ss_string, len(ss_string))

    ss_string = pdb_to_ss_string(pdb_path)
    print(ss_string, len(ss_string))

    bp = bp_from_ss_string(ss_string)
    print(bp)


