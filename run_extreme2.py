#!/usr/bin/env python
"""
Script for training model.
Use `run_extreme2.py -h` to see an auto-generated description of advanced options.
"""

import os
import sys
import argparse

import numpy

from extreme2.sequences import load_fasta_sequences

def get_args():
    parser = argparse.ArgumentParser(description="Train model.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', required=True,
                        help=('Input FASTA file'), type=str)
    parser.add_argument('-a', '--alph',
                        help=('Alphabet (default: dna)'),
                        type=str, choices=['dna', 'rna', 'protein'], default='dna')
    parser.add_argument('-p', '--pseudocount',
                        help=('Pseudocount to add to motifs (default: 0.0001).'),
                        type=float, default=0.0001)    
    parser.add_argument('-s', '--seed',
                        help=('Random seed for reproducibility (default: 1337).'),
                        type=int, default=1337)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    np.random.seed(args.seed)
    fasta_file = args.input
    pseudo_count = args.pseudocount
    seqs = load_fasta_sequences(fasta_file)

if __name__ == '__main__':
    main()

