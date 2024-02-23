### checked!
import numpy as np
from Bio import pairwise2


### checked!
########## Process PDB file ##########
def get_pdb_xyz(pdb_file):
    """ extract coordinate from .pdf text.
    
    backbone's(N, CA, C, O) xyz + R_group's centroid xyz.

    Args:
        pdb_file (list<string>): pdb text content.

    Return:
        X (np.ndarray): shape(AA_len, 5, 3)    
    """
    current_pos = -1000
    X = []
    current_aa = {} # N, CA, C, O, R
    for line in pdb_file:
        if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
            if current_aa != {}:
                R_group = []
                for atom in current_aa:
                    if atom not in ["N", "CA", "C", "O"]:
                        R_group.append(current_aa[atom])
                if R_group == []:
                    R_group = [current_aa["CA"]]
                R_group = np.array(R_group).mean(0)
                X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])
                current_aa = {}
            if line[0:4].strip() != "TER":
                current_pos = int(line[22:26].strip())

        if line[0:4].strip() == "ATOM":
            atom = line[13:16].strip()
            if atom != "H":
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                current_aa[atom] = xyz
    return np.array(X)


### checked!
########## Get DSSP ##########
def process_dssp(dssp_file):
    """ extract Second-Structure(SS) and relative-solvent-accessibility(RSA) from .dssp file.

    Args:
        dssp_file (string)

    Return:
        seq (string): AA sequence in .dssp file.
        dssp_feature (list<ndarray>): for each AA, use representation of (9,) array, 
        first element represent RSA, rest 8 elements represent SS_type in one-hot.
    """
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(8)
        SS_vec[SS_type.find(SS)] = 1
        ASA = float(lines[i][34:38].strip())
        RSA = min(1, ASA / rASA_std[aa_type.find(aa)]) # relative solvent accessibility
        dssp_feature.append(np.concatenate((np.array([RSA]), SS_vec)))

    return seq, dssp_feature


### checked!
def match_dssp(seq, dssp, ref_seq):
    """ pad dssp with np.zeros(9) if seq have gap according to ref_seq.
    
    Args:
        seq (string): dssp_seq.
        dssp (list<ndarray>)
        ref_seq: original sequence.

    Return:
        matched_dssp (list<ndarray>)
    """
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB

    padded_item = np.zeros(9)

    new_dssp = []
    for aa in seq:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp
