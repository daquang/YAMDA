#!/usr/bin/env python
"""
Script for erasing annoying sequences from FASTA files.
Use `erase_annoying_sequences.py -h` to see an auto-generated description of advanced options.
"""

import argparse
from yamda.sequences import load_fasta_sequences, save_fasta, erase_subsequences


def get_args():
    parser = argparse.ArgumentParser(description="Train model.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', required=True,
                        help='Input FASTA file', type=str)
    parser.add_argument('-o', '--output', default=None,
                        help='Output FASTA file of negative sequences', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    fasta_file = args.input
    output_file = args.output
    print('Loading sequences from FASTA')
    seqs, keys = load_fasta_sequences(fasta_file, return_keys=True)
    annoying_subsequences = [
        'AAAAAA',
        'TTTTTT',
        'CCCCGCCCC',
        'GGGGCGGGG'
    ]
    seqs = erase_subsequences(seqs, annoying_subsequences)
    save_fasta(output_file, seqs, keys)


if __name__ == '__main__':
    main()