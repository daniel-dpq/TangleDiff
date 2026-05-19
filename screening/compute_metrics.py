import pandas as pd
import os
import json
import numpy as np
from tanglediff.gln import pdb_to_core_gln_matrix
import freesasa
from Bio.PDB import PDBParser, PDBIO
import json
import mdtraj as md
from tqdm import tqdm
from Bio.PDB import Structure, Model, Chain, Residue, Atom
from multiprocessing import Pool

def analyze_interface(pdb_file, score_path, pack=True, analysis_binary='InterfaceAnalyzer.static.linuxgccrelease'):
    if pack:
        cmd = f'{analysis_binary} -in:file:s {pdb_file} -out:file:score_only {score_path} -pack_input -pack_separated'
    else:
        cmd = f'{analysis_binary} -in:file:s {pdb_file} -out:file:score_only {score_path}'
    os.system(cmd)


def parse_interface_sc(score_file):
    df = pd.read_csv(score_file, sep='\s+', skiprows=1)
    assert len(df) == 1
    return {'binding_energy': df['dG_separated'][0], 'BSA': df['dSASA_int'][0], '100BE/BSA': df['dG_separated/dSASAx100'][0]}


def get_ss_percentage(ss):
    return ss.count('H') / len(ss), ss.count('E') / len(ss), ss.count('C') / len(ss)


def ss2core_range(ss_string):
    H_start = ss_string.find('H')
    E_start = ss_string.find('E')
    if H_start == -1 and E_start == -1:
        start = 0
        end = 1
    elif H_start == -1:
        start = E_start
        end = ss_string.rfind('E')
    elif E_start == -1:
        start = H_start
        end = ss_string.rfind('H')
    else:
        start = min(H_start, E_start)
        end = max(ss_string.rfind('H'), ss_string.rfind('E'))
    assert start < end, f'start should be less than end, {start} {end} {ss_string}'
    return start, end


def get_core_terminal_dist(pdb_path, core_range):
    # parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_path)
    start, end = core_range

    # get the chains
    chains = list(structure.get_chains())
    assert len(chains) == 2, 'The number of chains is not 2'

    inter_dist_cn = []
    intra_dist = []

    # get the intra-chain terminal distance for each chain
    for chain in chains:
        residues = list(chain.get_residues())
        first_residue = residues[start]
        last_residue = residues[end]
        intra_dist.append(first_residue['CA'] - last_residue['CA'])

    # get the inter-chain terminal distance
    first_chain = chains[0]
    last_chain = chains[-1]
    first_residue = list(first_chain.get_residues())[start]
    last_residue = list(last_chain.get_residues())[end]
    inter_dist_cn.append(first_residue['CA'] - last_residue['CA'])

    first_residue = list(last_chain.get_residues())[start]
    last_residue = list(first_chain.get_residues())[end]
    inter_dist_cn.append(first_residue['CA'] - last_residue['CA'])

    # get other inter-chain terminal distance
    first_residue_1 = list(first_chain.get_residues())[start]
    first_residue_2 = list(last_chain.get_residues())[start]
    last_residue_1 = list(first_chain.get_residues())[end]
    last_residue_2 = list(last_chain.get_residues())[end]
    inter_dist_nn = first_residue_1['CA'] - first_residue_2['CA']
    inter_dist_cc = last_residue_1['CA'] - last_residue_2['CA']

    return intra_dist, inter_dist_cn, inter_dist_nn, inter_dist_cc


def get_terminal_dist(pdb_path):
    # parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_path)

    # get the chains
    chains = list(structure.get_chains())
    assert len(chains) == 2, 'The number of chains is not 2'

    inter_dist_cn = []
    intra_dist = []

    # get the intra-chain terminal distance for each chain
    for chain in chains:
        residues = list(chain.get_residues())
        first_residue = residues[0]
        last_residue = residues[-1]
        intra_dist.append(first_residue['CA'] - last_residue['CA'])

    # get the inter-chain terminal distance
    first_chain = chains[0]
    last_chain = chains[-1]
    first_residue = list(first_chain.get_residues())[0]
    last_residue = list(last_chain.get_residues())[-1]
    inter_dist_cn.append(first_residue['CA'] - last_residue['CA'])

    first_residue = list(last_chain.get_residues())[0]
    last_residue = list(first_chain.get_residues())[-1]
    inter_dist_cn.append(first_residue['CA'] - last_residue['CA'])

    # get other inter-chain terminal distance
    first_residue_1 = list(first_chain.get_residues())[0]
    first_residue_2 = list(last_chain.get_residues())[0]
    last_residue_1 = list(first_chain.get_residues())[-1]
    last_residue_2 = list(last_chain.get_residues())[-1]
    inter_dist_nn = first_residue_1['CA'] - first_residue_2['CA']
    inter_dist_cc = last_residue_1['CA'] - last_residue_2['CA']

    return intra_dist, inter_dist_cn, inter_dist_nn, inter_dist_cc


def get_b_factor_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    atoms = structure.get_atoms()
    b_factors = [atom.get_bfactor() for atom in atoms]
    b_factors = np.array(b_factors).mean()
    return b_factors


def get_bsa(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('-', pdb_path)
    result, _ = freesasa.calcBioPDB(structure)
    tot_asa = result.totalArea()

    mono_asa = 0
    chains = list(structure.get_chains())
    assert(len(chains) == 2)
    for chain in chains:
        io = PDBIO()
        io.set_structure(chain)
        io.save('tmp.pdb')
        structure = freesasa.Structure('tmp.pdb')
        result = freesasa.calc(structure)
        mono_asa += result.totalArea()

    bsa = (mono_asa - tot_asa)/2
    return bsa


def get_symmetry_features(pdb_dir, df, sym_dir):

    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Getting symmetry features'):

        acc_num = row['acc_num']
        sym_file = os.path.join(sym_dir, f'{acc_num}.json')
        pdb_file = os.path.join(pdb_dir, f'{acc_num}.pdb')
        if not os.path.exists(pdb_file):
            raise ValueError(f'{pdb_file} does not exist!')
        if not os.path.exists(sym_file):
            os.system(f'/home/puqing/screening/analysis/3_symmetry/ananas {pdb_file} --json {sym_file} -C 5')
        
        with open(sym_file, 'r') as f:
            sym = json.load(f)
        if sym is None:
            df.loc[i, 'symmetry'] = 'c1'
            df.loc[i, 'sym_rmsd'] = np.nan
            df.loc[i, 'sym_axis'] = np.nan
            df.loc[i, 'sym_center'] = np.nan
        else:
            sym = sym[0]
            df.loc[i, 'symmetry'] = sym['group']
            df.loc[i, 'sym_rmsd'] = sym['Average_RMSD']
            axis = sym['transforms'][0]['AXIS']
            center = sym['transforms'][0]['CENTER']
            assert len(axis) == 3 and len(center) == 3
            axis = f'{axis[0]}, {axis[1]}, {axis[2]}'
            center = f'{center[0]}, {center[1]}, {center[2]}'
            df.loc[i, 'sym_axis'] = axis
            df.loc[i, 'sym_center'] = center

    return df

def crop_pdb_single(pdb_file, dst_file, crop_range):
    '''
        crop_range: both start and end are inclusive
    '''
    if not os.path.exists(pdb_file):
        raise ValueError(f'{pdb_file} does not exist!')
    start, end = crop_range
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    chains = list(structure.get_chains())
    assert len(chains) == 1
    chain1 = chains[0]
    residues1 = list(chain1.get_residues())
    residues1 = residues1[start:end+1]
    
    structure = Structure.Structure("new_structure")
    model = Model.Model(0)
    chain1 = Chain.Chain("A")
    for residue in residues1:
        chain1.add(residue)
    model.add(chain1)
    structure.add(model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(dst_file)


def crop_pdb(pdb_file, dst_file, crop_range):
    '''
        crop_range: both start and end are inclusive
    '''
    if not os.path.exists(pdb_file):
        raise ValueError(f'{pdb_file} does not exist!')
    start, end = crop_range
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    chains = list(structure.get_chains())
    assert len(chains) == 2
    chain1 = chains[0]
    chain2 = chains[1]
    residues1 = list(chain1.get_residues())
    residues2 = list(chain2.get_residues())
    residues1 = residues1[start:end+1]
    residues2 = residues2[start:end+1]
    
    structure = Structure.Structure("new_structure")
    model = Model.Model(0)
    chain1 = Chain.Chain("A")
    chain2 = Chain.Chain("B")
    for residue in residues1:
        chain1.add(residue)
    for residue in residues2:
        chain2.add(residue)
    model.add(chain1)
    model.add(chain2)
    structure.add(model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(dst_file)


if __name__ == '__main__':

    cropped_pdb_dir = '/media/puqing/puqing/datasets/short_sequence/esmfold_uniref90_pdb_cropped'
    pdb_dir = '/media/puqing/puqing/datasets/short_sequence/esmfold_uniref90_pdb'
    sym_dir = '/media/puqing/puqing/datasets/short_sequence/symmetry_uniref90_pdb'
    sc_pack_dir = '/media/puqing/puqing/datasets/short_sequence/esmfold_uniref90_sc_pack'
    csv_file = '/media/puqing/puqing/datasets/short_sequence/Uniref90_l0-100_core.csv'
    save_file = '/media/puqing/puqing/datasets/short_sequence/Uniref90_l0-100_core.csv'
    os.makedirs(sym_dir, exist_ok=True)
    os.makedirs(sc_pack_dir, exist_ok=True)
    os.makedirs(cropped_pdb_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    df = df.dropna().reset_index(drop=True)
    df = df[(df['gln'].notna()) & (df['gln'].abs() >= 0.7)]
    df.drop_duplicates(subset=['acc_num'], inplace=True)
    print(f'number of entangled subunits: {len(df)}')
    input('Press any key to continue...')

    # get symmetry features
    df = get_symmetry_features(pdb_dir, df, sym_dir)

    # get secondary structure features and terminal distances
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Getting secondary structure features'):
        pdb_file = os.path.join(pdb_dir, f'{row["acc_num"]}.pdb')
        if not os.path.exists(pdb_file):
            raise ValueError(f'{pdb_file} does not exist!')
        # termial distance
        intra_dist_cn, inter_dist_nc, inter_dist_nn, inter_dist_cc = get_terminal_dist(pdb_file)
        df.loc[i, 'intra_dist'] = str(intra_dist_cn)
        df.loc[i, 'mean_intra_dist'] = np.mean(intra_dist_cn)
        df.loc[i, 'inter_dist'] = str(inter_dist_nc)
        df.loc[i, 'mean_inter_dist'] = np.mean(inter_dist_nc)
        df.loc[i, 'inter_dist_nn'] = inter_dist_nn
        df.loc[i, 'inter_dist_cc'] = inter_dist_cc

        # length
        df.loc[i, 'modeled_seq_len'] = len(row['sequence'])

        # secondary structure
        try:
            traj = md.load(pdb_file)
            pdb_ss = md.compute_dssp(traj, simplified=True)
            ss_string = ''.join(pdb_ss[0])
            monomer_length = len(row['sequence'])
            assert monomer_length * 2 == len(ss_string)
            df.loc[i, 'ss'] = ss_string[:monomer_length]
        except:
            df.loc[i, 'ss'] = np.nan
    
    df.dropna(inplace=True)
    df.to_csv(save_file, index=False)
    
    # core gln and terminal distance
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Getting core gln'):
        pdb_file = os.path.join(pdb_dir, f'{row["acc_num"]}.pdb')
        cropped_pdb_file = os.path.join(cropped_pdb_dir, f'{row["acc_num"]}.pdb')
        if not os.path.exists(pdb_file):
            raise ValueError(f'{pdb_file} does not exist!')
        core_gln = pdb_to_core_gln_matrix(pdb_file, 2)
        core_gln = core_gln.sum() if core_gln is not None else None
        df.loc[i, 'core_gln'] = core_gln

        core_range = ss2core_range(row['ss'])
        intra_dist_cn, inter_dist_nc, inter_dist_nn, inter_dist_cc = get_core_terminal_dist(pdb_file, core_range)
        df.loc[i, 'core_intra_dist'] = str(intra_dist_cn)
        df.loc[i, 'core_mean_intra_dist'] = np.mean(intra_dist_cn)
        df.loc[i, 'core_inter_dist'] = str(inter_dist_nc)
        df.loc[i, 'core_mean_inter_dist'] = np.mean(inter_dist_nc)
        df.loc[i, 'core_inter_dist_nn'] = inter_dist_nn
        df.loc[i, 'core_inter_dist_cc'] = inter_dist_cc
        df.loc[i, 'core_sequence'] = row['sequence'][core_range[0]:core_range[1]+1]
        df.loc[i, 'core_modeled_seq_len'] = len(df.loc[i, 'core_sequence'])
        df.loc[i, 'core_ss'] = row['ss'][core_range[0]:core_range[1]+1]
        ss_percentage = get_ss_percentage(df.loc[i, 'core_ss'])
        df.loc[i, 'core_H_percentage'] = ss_percentage[0]
        df.loc[i, 'core_E_percentage'] = ss_percentage[1]
        df.loc[i, 'core_C_percentage'] = ss_percentage[2]
        assert len(df.loc[i, 'core_sequence']) == len(df.loc[i, 'core_ss'])

        crop_pdb(pdb_file, cropped_pdb_file, core_range)
        core_plddt = get_b_factor_from_pdb(cropped_pdb_file)
        df.loc[i, 'core_plddt'] = core_plddt

    df.to_csv(save_file, index=False)

    # compute binding energy
    print('Computing binding energy...')
    cropped_pdb_files = os.listdir(cropped_pdb_dir)
    def compute_binding_energy(pdb_file):
        acc_num = pdb_file.split('.')[0]
        assert len(df[df['acc_num'] == acc_num]) == 1
        pdb_path = os.path.join(cropped_pdb_dir, pdb_file)
        sc_pack_path = os.path.join(sc_pack_dir, pdb_file.replace('.pdb', '_pack.sc'))
        if not os.path.exists(sc_pack_path):
            analyze_interface(pdb_path, sc_pack_path, pack=True)
    with Pool(16) as p:
        p.map(compute_binding_energy, cropped_pdb_files)

    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Getting binding energy'):
        pdb_file = os.path.join(cropped_pdb_dir, f'{row["acc_num"]}.pdb')
        sc_pack_file = os.path.join(sc_pack_dir, f'{row["acc_num"]}_pack.sc')
        if not os.path.exists(sc_pack_file):
            raise ValueError(f'{sc_pack_file} does not exist!')
        result = parse_interface_sc(sc_pack_file)
        df.loc[i, 'core_binding_energy_pack'] = result['binding_energy']
        df.loc[i, 'core_BSA'] = result['BSA']
    
    df.to_csv(save_file, index=False)

    df_cluster = pd.read_csv('Uniref90_l0-100_seq_unique.csv')
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Getting cluster features'):
        acc_num = row['acc_num']
        assert len(df_cluster[df_cluster['acc_num'] == acc_num]) == 1, f'{len(df_cluster[df_cluster["acc_num"] == acc_num])}, {acc_num}'
        cluster = df_cluster[df_cluster['acc_num'] == acc_num]['cluster50'].values[0]
        df.loc[i, 'cluster50'] = cluster
    
    df.to_csv(save_file, index=False)

    # concatenate the two dataframes
    df1 = pd.read_csv(save_file)
    df2 = pd.read_csv('uniref50_l50-100_core_cluster_be.csv')

    # find the common and unique columns
    common_columns = list(set(df1.columns).intersection(set(df2.columns)))
    unique_columns_1 = list(set(df1.columns).difference(set(df2.columns)))
    unique_columns_2 = list(set(df2.columns).difference(set(df1.columns)))
    print(f'common columns: {common_columns}')
    print(f'unique columns in df1: {unique_columns_1}')
    print(f'unique columns in df2: {unique_columns_2}')
    input('Press any key to continue...')

    # merge the two dataframes
    df1 = df1[common_columns]
    df2 = df2[common_columns]
    df = pd.concat([df1, df2], axis=0)
    df = df.dropna().reset_index(drop=True)

    # save the merged dataframe
    df.to_csv('Uniref_final.csv', index=False)


    df = pd.read_csv('Uniref_final_filtered_relax_be_os.csv')
    print(f'number of entangled subunits: {len(df)}')

    cluster_df = pd.read_csv('Uniref_final_filtered_clust30_cluster.tsv', sep='\t', header=None, names=['rep', 'member'])
    print(cluster_df.head())
    print(cluster_df['rep'].nunique())
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Getting cluster features'):
        acc_num = row['acc_num']

        assert len(cluster_df[cluster_df['member'] == acc_num]) == 1, f'{len(cluster_df[cluster_df["member"] == acc_num])}, {acc_num}'
        cluster = cluster_df[cluster_df['member'] == acc_num]['rep'].values[0]
        df.loc[i, 'cluster30'] = cluster
    
    df.to_csv('Uniref_final_filtered_relax_be_os_clust30.csv', index=False)

    