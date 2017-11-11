import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange

from . import sequences


class TCM:
    def __init__(self, n_seeds, motif_width, min_sites,
                 batch_size, cuda=True, init='subsequences'):
        self.n_seeds = n_seeds
        self.motif_width = motif_width
        self.min_sites = min_sites
        self.batch_size = batch_size
        self.cuda = cuda
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
        N = len(X)
        min_sites = self.min_sites
        max_sites = N # Expect at most one motif occurrence per sequence
        # Extract valid one-hot subsequences
        X = sequences.get_subsequences(X, self.motif_width)
        M, L, W = X.shape
        # Compute background frequencies
        letter_frequency = X.sum(axis=(0,2))
        bg_probs = 1.0 * letter_frequency / letter_frequency.sum()
        ppms_bg_seeds = bg_probs.reshape([1, L, 1]).repeat(
                            self.n_seeds, axis=0).astype(np.float32)
        ppms_bg_seeds = torch.from_numpy(ppms_bg_seeds)
        # Initialize PPMs
        if self.init == 'dirichlet':
            ppms_seeds = np.random.dirichlet(
                         self.bg_probs, size=(self.n_seeds, self.motif_width))
        elif self.init == 'subsequences':
            ppms_seeds = X[0:self.n_seeds].astype(np.float32) * 0.7
            ppms_seeds[ppms_seeds==0] = 0.1
            ppms_seeds = torch.from_numpy(ppms_seeds)
        else:
            sys.exit(1)
        if self.cuda:
            ppms_bg_seeds = ppms_bg_seeds.cuda()
            ppms_seeds = ppms_seeds.cuda()
        ppms_bg_seeds = ppms_bg_seeds.expand(self.n_seeds, L, W)
        sites_i = max_sites
        while sites_i <= max_sites:
            fracs_seeds = torch.ones(self.n_seeds) * sites_i / M
            if self.cuda:
                fracs_seeds = fracs_seeds.cuda()
            ppms_seeds_i, ppms_bg_seeds_i, fracs_seeds_i, ll_i = \
                self._batch_em(X, ppms_seeds, ppms_bg_seeds, fracs_seeds, 1)
            sites_i *= 2
        return self

    def transform(self, X):
        return

    def _batch_em(self, X, ppms, ppms_bg, fracs, epochs):
        M, L, W = X.shape
        n_filters = len(ppms)
        m = nn.Conv1d(L, n_filters, W, bias=False)
        m.weight.data = torch.log(ppms_bg) - torch.log(ppms)
        fracs = fracs.view((1, n_filters, 1))
        pfms = torch.zeros((n_filters, L, W))
        pfms_bg = torch.zeros((n_filters, L, W))
        counts = torch.zeros((n_filters,1))
        if self.cuda:
            m.cuda()
            pfms = pfms.cuda()
            pfms_bg = pfms_bg.cuda()
            counts = counts.cuda()
        for i in range(epochs):
            # E-step, compute membership weights and letter frequencies
            pfms.zero_()
            pfms_bg.zero_()
            counts.zero_()
            fracs_ratio = (1 - fracs) / fracs
            for j in trange(0, M, self.batch_size):
                batch = X[j:j+self.batch_size]
                batch_size = len(batch)
                x = Variable(torch.from_numpy(batch).float())
                if self.cuda:
                    x = x.cuda()
                log_ratios = m(x).data
                ratios = torch.exp(log_ratios)
                #state_probs = fracs / (fracs + (1 - fracs) * ratios)
                #state_probs = 1 / (1 + (1 / fracs - 1) * ratios)
                state_probs = 1 / (1 + fracs_ratio * ratios)
                counts.add_(state_probs.sum(dim=0))
                pfms.add_((state_probs.unsqueeze(-1) *
                           x.data.unsqueeze(1)).sum(dim=0))
                pfms_bg.add_(((1 - state_probs.unsqueeze(-1)) *
                               x.data.unsqueeze(1)).sum(dim=0))
            # M-step, update parameters
            fracs = (counts / M).unsqueeze(0)
            ppms = pfms / counts.unsqueeze(2)
            ppms_bg = (pfms_bg.sum(dim=-1) /
                       (W * (M - counts))).unsqueeze(2).expand(n_filters, L, W)
            log_likelihoods = self._compute_log_likelihood(X, ppms, ppms_bg, fracs)
        fracs = fracs.view(-1)
        return ppms, ppms_bg, fracs, log_likelihoods

    def _compute_log_likelihood(self, X, ppms, ppms_bg, fracs):
        M, L, W = X.shape
        n_filters = len(ppms)
        m_ppms = nn.Conv1d(L, n_filters, W, bias=False)
        m_ppms.weight.data = torch.log(ppms)
        m_ppms_bg = nn.Conv1d(L, n_filters, W, bias=False)
        m_ppms_bg.weight.data = torch.log(ppms_bg)
        log_likelihoods = torch.zeros(n_filters)
        if self.cuda:
            m_ppms.cuda()
            m_ppms_bg.cuda()
            log_likelihoods = log_likelihoods.cuda()
        for j in trange(0, M, self.batch_size):
            batch = X[j:j+self.batch_size]
            x = Variable(torch.from_numpy(batch).float())
            if self.cuda:
                x = x.cuda()
            ppms_prob = torch.exp(m_ppms(x)).data
            ppms_prob_bg = torch.exp(m_ppms_bg(x)).data
            log_likelihoods.add_(torch.log(fracs*ppms_prob + (1-fracs)*ppms_prob_bg).sum(dim=0).view(-1))
        return log_likelihoods

    def _online_e_step(self):
        return

    def _online_m_step(self):
        #ct is (10, 10, 4) samples tensor
        #at is (5, 10, 4) motifs tensor
        #dt = torch.mm(ct.view(10,40),at.view(5,40).transpose(1,0))
        return

    def sample(self, n_samples=1):
        return
