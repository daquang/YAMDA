import sys

import numpy as np
import torch
import torch.nn as nn

import initialize
import sequences

class TCM:
    def __init__(self, n_motifs, motif_width, alpha='dna', init='subsequences'):
        self.n_motifs = n_motifs
        self.motif_width = motif_width
        self.alpha = alpha
        self.init = init

    def _initialize_parameters(self, X):
        """Initialize the model parameters.
        Parameters
        ----------
        X : numpy array, shape  (n_samples, width, n_letters)
        """
        N, W, L = X.shape
        self.L = L
        self.W = W
        letter_frequency = X.sum(axis=(0,1))
        bg_probs = 1.0 * letter_frequency / letter_frequency.sum()
        self.ppm_bg = bg_probs.reshape([1, L]).repeat(self.n_motifs, axis=0)

    def fit(self, X):
        """Fit the model to the data X.
        Parameters
        ----------
        X : {list of numpy char arrays}
            Training data.
        Returns
        -------
        self : TCM
            The fitted model.
        """
        # Extract valid one-hot subsequences
        X_subseqs_onehot = sequences.get_onehot_subsequence(X, self.motif_width, self.alpha)
        # Compute background frequencies
        letter_frequency = X_subseqs_onehot.sum(axis=(0,2))
        bg_probs = 1.0 * letter_frequency / letter_frequency.sum()
        self.ppms_bg = bg_probs.reshape([len(bg_probs), 1]).repeat(self.n_motifs, axis=0)
        # Initialize PPMs
        if self.init == 'dirichlet':
            self.ppms = np.random.dirichlet(self.bg_probs, size=(self.n_motifs, self.motif_width))
        elif self.init == 'subsequences'
            self.ppms = X_subseqs_onehot[0:self.n_motifs].astype(np.float32) * 0.7
            self.ppms[self.ppms==0] = 0.1
        else
            sys.exit(1)
        # Create torch model
        conv = nn.Conv1d(len(letter_frequency), self.n_motifs, self.motif_width)
        conv.weight.data = self.ppms
        # Initialize fractions
        return self

    def transform(self, X):
        return

    def _online_e_step(self):
        return

    def _online_m_step(self):
        #ct is (10, 10, 4) samples tensor
        #at is (5, 10, 4) motifs tensor
        #dt = torch.mm(ct.view(10,40),at.view(5,40).transpose(1,0))
        return

    def sample(self, n_samples=1):
        return

