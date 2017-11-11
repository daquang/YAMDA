#!/usr/bin/env python

import numpy as np

import sys
import os


def load_model():
    return


def save_model(dirname, model, clobber):
    try:
        os.makedirs(dirname)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            if not clobber:
                print(('output directory (%s) already exists '
                       'but you specified not to clobber it') % dirname)
                sys.exit(1)
            else:
                print(('output directory (%s) already exists '
                       'so it will be clobbered') % dirname)
    return


def save_meme(fname, ppms, nsites=None, alpha='dna'):
    if alpha == 'dna':
        d = np.array(['A', 'C', 'G', 'T'])
        strands_str = 'strands: + -\n\n'
    elif alpha == 'rna':
        d = np.array(['A', 'C', 'G', 'U'])
        strands_str = 'strands: + -\n\n'
    elif alpha == 'protein':
        d = np.array(['A', 'C', 'D', 'E',
                      'F', 'G', 'H', 'I',
                      'K', 'L', 'M', 'N',
                      'P', 'Q', 'R', 'S',
                      'T', 'V', 'W', 'Y'])
        strands_str = 'strands: + \n\n'
    else:
        sys.exit(1)

    if nsites is not None:
        assert len(nsites) == len(ppms)
    else:
        nsites = len(ppms) * [1337]

    f = open(fname, 'w')

    alength = len(d)
    alphabet_str = 'ALPHABET= ' + ''.join(d) + '\n\n'
    freq_str = ''.join([a + ' %f ' % (1.0/alength) for a in d]) + '\n\n'
    header = 'MEME version 4\n\n' + \
              alphabet_str + \
              strands_str + \
             'Background letter frequencies (from uniform background):\n' + \
             freq_str

    f.write(header)

    for i, ppm in enumerate(ppms):
        w = ppm.shape[1]
        motif_header = 'MOTIF M%i N%i' % (i, i) + '\n\n' + \
                       'letter-probability matrix: ' + \
                       'alength= %i w= %i nsites= %i E= 0\n' % (alength, w, nsites[i])
        f.write(motif_header)

        ppm_str = ''
        for j in range(w):
            ppm_str += '%f %f %f %f\n' % tuple(1.0*ppm[:,j]/ppm[:,j].sum())
        ppm_str += '\n'
        f.write(ppm_str)

    f.close()


def load_meme(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    num_lines = len(lines)
    i = 0
    ppms = []
    identifiers = []
    names = []
    nsites_list = []
    d = None
    while i < num_lines:
        line = lines[i]
        if 'ALPHABET' in line:
            alpha_str = line.split()[-1].strip()
            d = np.array(list(alpha_str))
        if 'MOTIF' in line:
            name_info = line.split()
            identifier = name_info[1]
            name = name_info[2]
            while 'letter-probability matrix' not in line:
                i += 1
                line = lines[i]
            motif_info = lines[i]
            motif_info = motif_info.split()
            w_index = motif_info.index('w=') + 1
            w = int(motif_info[w_index])
            nsites_index = motif_info.index('nsites=') + 1
            nsites = int(motif_info[nsites_index])
            motif = np.zeros((len(d), w))
            i += 1
            line = lines[i]
            while len(line.strip()) == 0:
                i += 1
                line = lines[i]
            for j in range(w):
                motif[:, j] = np.array(lines[i].split(), dtype=float)
                i += 1
            ppm = np.dot(motif, np.diag(1/motif.sum(axis=0)))
            ppms.append(ppm)
            nsites_list.append(nsites)
            names.append(name)
            identifiers.append(identifier)
        i += 1
    return ppms, d, names, identifiers, nsites_list

