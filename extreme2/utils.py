#!/usr/bin/env python

import numpy as np

import sys
import os

def load_model():
    return

def save_model():
    return

def save_meme(fname, ppms, alpha='dna'):
    if alpha=='dna':
        d = np.array(['A','C','G','T'])
    elif alpha=='rna':
        d = np.array(['A','C','G','U'])
    elif alpha=='protein':
        d = np.array(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
    else:
        sys.exit(1)

    alpha_len = len(d)
    alphabet_str = 'ALPHABET= ' + ' '.join(d) + '\n\n'
    freq_str = 
    header = 'MEME version 4\n\n' +
              alphabet_str +
              strands_str +
             'Background letter frequencies (from uniform background):\n' +
             freq_str'

def load_meme(fname):
    return


