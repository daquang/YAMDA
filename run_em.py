#!/usr/bin/env python3
"""
Script for training model.
Use `run_em.py -h` to see an auto-generated description of advanced options.
"""

import argparse

import numpy as np
import torch

from yamda.sequences import load_fasta_sequences, save_fasta
from yamda.mixture import TCM
from yamda.utils import save_model


def get_args():
    parser = argparse.ArgumentParser(description="Train model.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', required=True,
                        help='Input FASTA file', type=str)
    parser.add_argument('-j', '--input2', default=None,
                        help='Input FASTA file of negative sequences', type=str)
    parser.add_argument('-b', '--batchsize', type=int, default=1000,
                        help='Input batch size for training (default: 1000)')
    parser.add_argument('-a', '--alpha',
                        help='Alphabet (default: dna)',
                        type=str, choices=['dna', 'rna', 'protein'], default='dna')
    parser.add_argument('-r', '--revcomp', action='store_true', default=False,
                        help='Consider both the given strand and the reverse complement strand when searching for '
                             'motifs in a complementable alphabet (default: consider given strand only).')
    parser.add_argument('-m', '--model',
                        help='Model (default: tcm)',
                        type=str, choices=['tcm', 'zoops', 'oops'], default='tcm')
    parser.add_argument('-e', '--erasewhole', action='store_true', default=False,
                        help='Erase an entire sequence if it contains a discovered motif '
                             '(default: erase individual motif occurrences).')
    parser.add_argument('-f', '--fudge',
                        help='Fudge factor to help with extremely rare motifs. Should be >0 and <=1 (default: 0.1).',
                        type=float, default=0.1)
    parser.add_argument('-w', '--width',
                        help='Motif width (default: 20).',
                        type=int, default=20)
    parser.add_argument('-k', '--halflength',
                        help='k-mer half-length for gapped k-mer search seeding (default: 6).',
                        type=int, default=6)
    parser.add_argument('-n', '--nmotifs',
                        help='Number of motifs to find (default: 1).',
                        type=int, default=1)
    parser.add_argument('-mins', '--minsites',
                        help='Minimum number of motif occurrences (default: 100).',
                        type=int, default=100)
    parser.add_argument('-maxs', '--maxsites',
                        help='Maximum number of motif occurrences. If left unspecified, will default to number of'
                             'sequences.',
                        type=int, default=None)
    parser.add_argument('-ns', '--nseeds',
                        help='Number of motif seeds to try. If left unspecified, will default to 100 (for gapped k-mer'
                             'search) or 1000 (for randomly sampled subsequence method).',
                        type=int, default=None)
    parser.add_argument('-maxiter', '--maxiter',
                        help='Maximum number of refining iterations of batch EM to run from any starting '
                             'point. Batch EM is run for maxiter iterations or until convergence (see '
                             '-tolerance, below) from each starting point for refining (default: 20)',
                        type=int, default=20)
    parser.add_argument('-t', '--tolerance',
                        help='Stop iterating refining batch/on-line EM when the change in the motif probability matrix '
                             'is less than tolerance. Change is defined as the euclidean distance between two '
                             'successive frequency matrices (default: 1e-3).',
                        type=float, default=1e-3)
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disable the default CUDA training.')
    parser.add_argument('-s', '--seed',
                        help='Random seed for reproducibility (default: 1337).',
                        type=int, default=1337)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-o', '--outputdir', type=str,
                       help='The output directory. Causes error if the directory already exists.')
    group.add_argument('-oc', '--outputdirc', type=str,
                       help='The output directory. Will overwrite if directory already exists.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    fasta_file = args.input
    neg_fasta_file = args.input2
    alpha = args.alpha
    revcomp = args.revcomp
    if alpha == 'protein' and revcomp:
        revcomp = False
        print('You specified reverse complement, but proteins lack reverse complements!')
    fudge = args.fudge
    assert 0 < fudge <= 1
    half_length = args.halflength
    motif_width = args.width
    if 2 * half_length > motif_width:
        print('The half length, %i, is too big for the motif width, %i. Changing half length to %i' % (half_length,
                                                                                                       motif_width,
                                                                                                       motif_width//2))
        half_length = motif_width // 2
    if args.model != 'tcm':
        print('Only the TCM/ANR model is currently available.')
    min_sites = args.minsites
    assert min_sites > 0
    max_sites = args.maxsites
    assert max_sites is None or max_sites >= min_sites
    batch_size = args.batchsize
    erasewhole = args.erasewhole
    tolerance = args.tolerance
    maxiter = args.maxiter
    n_seeds = args.nseeds
    n_motifs = args.nmotifs
    print('Loading sequences from FASTA')
    seqs = load_fasta_sequences(fasta_file)
    if neg_fasta_file is None:
        seqs_neg = None
        if n_seeds is None:
            n_seeds = 1000
    else:
        seqs_neg = load_fasta_sequences(neg_fasta_file)
        if n_seeds is None:
            n_seeds = 100
    model = TCM(n_seeds, n_motifs, motif_width, min_sites, max_sites, batch_size, half_length, fudge, alpha, revcomp,
                tolerance, maxiter, erasewhole, cuda)
    seqs, seqs_neg = model.fit(seqs, seqs_neg)
    if args.outputdir is None:
        overwrite = True
        output_dir = args.outputdirc
    else:
        overwrite = False
        output_dir = args.outputdir
    print('Saving results to ' + output_dir)
    save_model(output_dir, model, overwrite)
    save_fasta(output_dir + '/positive_seqs.fa', seqs)
    if neg_fasta_file is not None:
        save_fasta(output_dir + '/negative_seqs.fa', seqs_neg)


if __name__ == '__main__':
    main()
