import sys

import numpy as np
import torch
import torch.nn as nn

import sequences

class TCM:
    def __init__(self, n_seeds, motif_width, cuda=True, init='subsequences'):
        self.n_seeds = n_seeds
        self.motif_width = motif_width
        self.init = init

    def fit(self, X):
        """Fit the model to the data X. Discover one motif.
        Parameters
        ----------
        X : {list of one hot encoded numpy arrays}
            Training data.
        Returns
        -------
        self : TCM
            The fitted model.
        """
        # Extract valid one-hot subsequences
        X = sequences.get_subsequences(X, self.motif_width)
        N, L, W = X.shape
        # Compute background frequencies
        letter_frequency = X.sum(axis=(0,2))
        bg_probs = 1.0 * letter_frequency / letter_frequency.sum()
        ppms_bg = bg_probs.reshape([len(bg_probs), 1]).repeat(self.n_seeds, axis=0)
        # Initialize PPMs
        if self.init == 'dirichlet':
            ppms = np.random.dirichlet(self.bg_probs, size=(self.n_seeds, self.motif_width))
        elif self.init == 'subsequences':
            ppms = X[0:self.n_seeds].astype(np.float32) * 0.7
            ppms[ppms==0] = 0.1
        else
            sys.exit(1)
        # Create torch model
        conv = nn.Conv1d(L, self.n_seeds, W, bias=False)
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

