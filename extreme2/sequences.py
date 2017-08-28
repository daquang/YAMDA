import sys

from itertools import chain
import numpy as np
import pyfasta


def load_fasta_sequences(fasta_file):
    """
    Reads a FASTA file and returns letter sequences as list of numpy arrays
    """
    fasta = pyfasta.Fasta(fasta_file)
    seqs = [np.array(fasta[k]) for k in fasta.iterkeys()]
    return seqs

def get_onehot_subsequence(seqs, W, alpha='dna'):
    """
    Extracts one hot coded subsequences from list of numpy array letter sequences.
    Filters away subsequences overlapping invalid letters.
    """
    if alpha=='dna':
        d = np.array(['A','C','G','T'])
    elif alpha=='rna':
        d = np.array(['A','C','G','U'])
    elif alpha=='protein':
        d = np.array(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
    else:
        sys.exit(1)
    seqs_onehot = [seq[:, np.newaxis]==d for seq in seqs]
    subseqs_onehot = [[seq_onehot[i:i+W] for i in range(len(seq_onehot)-W+1)] 
                      for seq_onehot in seqs_onehot]
    subseqs_onehot = list(chain(*subseqs_onehot))
    subseqs_onehot = np.array(subseqs_onehot)
    # filter subsequences containing invalid letters
    subseqs_onehot = subseqs_onehot[subseqs_onehot.sum(axis=(-1,-2))==W]
    return subseqs_onehot

