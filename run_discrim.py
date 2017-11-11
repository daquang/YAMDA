#!/usr/bin/env python
"""
Script for finding motifs with a neural network model.
Use `run_discrim.py -h` to see an auto-generated description of advanced options.
"""

import os
import sys
import argparse

import numpy as np
import torch
from yamda.sequences import load_fasta_sequences
from yamda.discrim import SeqDiscrim

def get_args():
    parser = argparse.ArgumentParser(description="Train model.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', required=True,
                        help='Input FASTA file', type=str)
    parser.add_argument('-j', '--input2', required=True,
                        help='Input FASTA file', type=str)
    parser.add_argument('--bed', action='store_true', default=False,
                        help='Input files are BED files instead of FASTA.')
    parser.add_argument('-r', '--revcomp', action='store_true', default=False,
                        help='Consider both the given strand and the reverse complement strand when searching for '
                             'motifs in a complementable alphabet (default: consider given strand only).')
    parser.add_argument('-b', '--batch-size', type=int, default=100,
                        help='Input batch size for training (default: 100)')
    parser.add_argument('-a', '--alpha',
                        help='Alphabet (default: dna)',
                        type=str, choices=['dna', 'rna', 'protein'], default='dna')
    parser.add_argument('-p', '--pseudocount',
                        help='Pseudocount to prevent arithmetic underflow (default: 0.0001).',
                        type=float, default=0.0001)
    parser.add_argument('-w', '--width',
                        help='Motif width (default: 20).',
                        type=int, default=20)
    parser.add_argument('-n', '--nmotif',
                        help='Number of motifs to find (default: 1).',
                        type=int, default=1)
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
    alpha = args.alpha
    revcomp = args.revcomp
    fasta_file = args.input
    fasta_file2 = args.input2
    pos_seqs = load_fasta_sequences(fasta_file, alpha)
    neg_seqs = load_fasta_sequences(fasta_file2, alpha)
    model = SeqDiscrim()
    model.fit(pos_seqs, neg_seqs)


if __name__ == '__main__':
    main()

