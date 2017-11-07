import sys

from itertools import chain
import numpy as np
from pyfaidx import Fasta


def load_fasta_sequences(fasta_file, alpha='dna'):
    """
    Reads a FASTA file and returns one-hot coded sequences as list of numpy arrays
    """
    if alpha=='dna':
        d = np.array(['A','C','G','T'])
    elif alpha=='rna':
        d = np.array(['A','C','G','U'])
    elif alpha=='protein':
        d = np.array(['A','C','D','E',
                      'F','G','H','I',
                      'K','L','M','N',
                      'P','Q','R','S',
                      'T','V','W','Y'])
    else:
        sys.exit(1)
    fasta = Fasta(fasta_file, as_raw=True, sequence_always_upper=True)
    seqs = [(d[:, np.newaxis] == np.array(list(seq[:]))).astype(np.uint8) for seq in fasta]
    return seqs


def get_subsequences(seqs, W):
    """
    Extracts W-length subsequences from list of one-hot coded
    sequences.
    Filters away subsequences overlapping invalid letters.
    """
    subseqs = [[seq[:,i:i+W] for i in range(seq.shape[1]-W+1)] 
                for seq in seqs]
    subseqs = list(chain(*subseqs))
    subseqs = np.array(subseqs, dtype=np.uint8)
    # filter away subsequences containing invalid letters
    subseqs = subseqs[subseqs.sum(axis=(-1,-2))==W]
    return subseqs


def pad_sequences(sequences, maxlen):
    L = len(sequences[0])
    num_samples = len(sequences)

    x = np.zeros((num_samples, L, maxlen), dtype=np.uint8)
    for idx, s in enumerate(sequences):
        x[idx, :, :s.shape[1]] = s
    return x

