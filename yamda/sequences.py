import sys

from itertools import chain
import numpy as np
from pyfaidx import Fasta
from tqdm import tqdm
import re

def load_fasta_sequences(fasta_file, return_keys=False):
    """
    Reads a FASTA file and returns list of string sequences
    """
    fasta = Fasta(fasta_file, as_raw=True, sequence_always_upper=True)
    seqs = [seq[:] for seq in fasta]
    if return_keys:
        keys = list(fasta.keys())
    fasta.close()
    if return_keys:
        return seqs, keys
    return seqs


def save_fasta(fname, seqs, seq_names=None):
    seqs_file = open(fname, 'w')
    for i in range(len(seqs)):
        if seq_names is None:
            seq_name = 'sequence' + str(i)
        else:
            seq_name = seq_names[i]
        seqs_file.write('>' + seq_name + '\n')
        seqs_file.write(seqs[i] + '\n')
    seqs_file.close()


def encode(seqs, alpha='dna'):
    "One-hot encode a list of strings as list of numpy arrays"
    if alpha == 'dna':
        d = np.array(['A', 'C', 'G', 'T'])
    elif alpha == 'rna':
        d = np.array(['A', 'C', 'G', 'U'])
    elif alpha == 'protein':
        d = np.array(['A', 'C', 'D', 'E',
                      'F', 'G', 'H', 'I',
                      'K', 'L', 'M', 'N',
                      'P', 'Q', 'R', 'S',
                      'T', 'V', 'W', 'Y'])
    else:
        sys.exit(1)
    seqs = [(d[:, np.newaxis] == np.array(list(seq))).astype(np.uint8)
            for seq in seqs]
    return seqs


def decode(seqs, alpha='dna'):
    if alpha == 'dna':
        d = np.array(['N', 'A', 'C', 'G', 'T'])
    elif alpha == 'rna':
        d = np.array(['N', 'A', 'C', 'G', 'U'])
    elif alpha == 'protein':
        d = np.array(['X', 'A', 'C', 'D', 'E',
                           'F', 'G', 'H', 'I',
                           'K', 'L', 'M', 'N',
                           'P', 'Q', 'R', 'S',
                           'T', 'V', 'W', 'Y'])
    seqs = [''.join(d[seq.max(axis=0) + seq.argmax(axis=0)]) for seq in seqs]
    return seqs

def get_onehot_subsequences(seqs, W):
    """
    Extracts W-length subsequences from list of one-hot coded
    numpy sequences.
    Filters away subsequences overlapping invalid letters.
    """
    subseqs = [[seq[:,i:i+W] for i in range(seq.shape[1]-W+1)] 
                for seq in seqs]
    subseqs = list(chain(*subseqs))
    subseqs = np.array(subseqs, dtype=np.uint8)
    # filter away subsequences containing invalid letters
    subseqs = subseqs[subseqs.sum(axis=(-1,-2))==W]
    return subseqs


def pad_onehot_sequences(seqs, maxlen, center=True):
    L = len(seqs[0])
    num_samples = len(seqs)
    x = np.zeros((num_samples, L, maxlen), dtype=np.uint8)
    for idx, s in enumerate(seqs):
        if center:
            start = int(maxlen / 2 - s.shape[1] / 2)
            stop = start + s.shape[1]
        else:
            start = 0
            stop = s.shape[1]
        x[idx, :, start:stop] = s
    return x


def erase_subsequences(seqs, annoying_subsequences):
    new_seqs = []
    annoying_subsequences_listed = [list(s) for s in annoying_subsequences]
    for seq in tqdm(seqs, desc='Erasing annoying sequences'):
        seq_l = list(seq)
        for s in annoying_subsequences:
            slen = len(s)
            replace_s = ['N'] * slen
            for m in re.finditer('(?=' + s + ')', seq):
                start = m.start()
                seq_l[start:start+slen] = replace_s
        new_seq = ''.join(seq_l)
        new_seqs.append(new_seq)
    return new_seqs


def get_rc(re):
    """
    Return the reverse complement of a DNA/RNA RE.
    """
    return re.translate(str.maketrans('ACGTURYKMBVDHSWN', 'TGCAAYRMKVBHDSWN'))[::-1]