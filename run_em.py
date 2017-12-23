#!/usr/bin/env python
"""
Script for training model.
Use `run_extreme2.py -h` to see an auto-generated description of advanced options.
"""

import os
import sys
import argparse

import numpy as np
import torch

from yamda.sequences import load_fasta_sequences
from yamda.mixture import TCM

def get_args():
    parser = argparse.ArgumentParser(description="Train model.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', required=True,
                        help='Input FASTA file', type=str)
    parser.add_argument('-b', '--batchsize', type=int, default=100,
                        help='Input batch size for training (default: 100)')
    parser.add_argument('-a', '--alpha',
                        help='Alphabet (default: dna)',
                        type=str, choices=['dna', 'rna', 'protein'], default='dna')
    parser.add_argument('-r', '--revcomp', action='store_true', default=False,
                        help='Consider both the given strand and the reverse complement strand when searching for '
                             'motifs in a complementable alphabet (default: consider given strand only).')
    parser.add_argument('-m', '--model',
                        help='Model (default: tcm)',
                        type=str, choices=['tcm', 'zoops', 'oops'], default='tcm')
    parser.add_argument('-p', '--pseudocount',
                        help='Pseudocount to prevent arithmetic underflow (default: 0.0001).',
                        type=float, default=0.0001)
    parser.add_argument('-w', '--width',
                        help='Motif width (default: 25).',
                        type=int, default=25)
    parser.add_argument('-n', '--nmotifs',
                        help='Number of motifs to find (default: 1).',
                        type=int, default=1)
    parser.add_argument('-mins', '--minsites',
                        help='Minimum number of motif occurrences (default: 100).',
                        type=int, default=100)
    parser.add_argument('-ns', '--nseeds',
                        help='Number of motif seeds to try (default: 1000).',
                        type=int, default=1000)
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disable the default CUDA training.')
    parser.add_argument('-s', '--seed',
                        help='Random seed for reproducibility (default: 1337).',
                        type=int, default=1337)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    np.random.seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    fasta_file = args.input
    alpha = args.alpha
    revcomp = args.revcomp
    pseudo_count = args.pseudocount
    motif_width = args.width
    min_sites = args.minsites
    batch_size = args.batchsize
    n_seeds = args.nseeds
    seqs = load_fasta_sequences(fasta_file)
    model = TCM(n_seeds, motif_width, min_sites, batch_size, alpha, revcomp,
                init='subsequences', cuda=cuda)
    model.fit(seqs)

if __name__ == '__main__':
    main()

