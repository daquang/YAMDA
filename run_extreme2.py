#!/usr/bin/env python
"""
Script for training model.
Use `run_extreme2.py -h` to see an auto-generated description of advanced options.
"""

import os
import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train model.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-a', '--alph',
                        help=('Alphabet (default: dna)'),
                        type=str, choices=['dna', 'rna', 'protein'], default='dna')
    parser.add_argument('-s', '--seed',
                        help=('Random seed for reproducibility (default: 1337).'),
                        type=int, default=1337)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

if __name__ == '__main__':
    main()

