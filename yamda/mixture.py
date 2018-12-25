import numpy as np
from scipy import signal
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange

from . import sequences
from . import initialize


class TCM:
    def __init__(self, n_seeds, n_motifs, motif_width, min_sites, max_sites, batch_size, half_length, fudge, alpha,
                 revcomp, tolerance, maxiter, erasewhole, cuda):
        self.n_seeds = n_seeds
        self.n_motifs = n_motifs
        self.motif_width = motif_width
        self.min_sites = min_sites
        self.max_sites = max_sites
        self.batch_size = batch_size
        self.half_length = half_length
        self.fudge = fudge
        self.alpha = alpha
        self.revcomp = revcomp
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.erasewhole = erasewhole
        self.cuda = cuda
        self.ppms_ = None
        self.ppms_bg_ = None
        self.fracs_ = None
        self.n_sites_ = None

    def fit(self, X, X_neg=None):
        """Fit the model to the data X. Discover n_motifs motifs.
        Parameters
        ----------
        X : {list of string sequences}
            Training data.
        Returns
        -------
        self : TCM
            The fitted model.
        """
        ppms_final = []
        ppms_bg_final = []
        fracs_final = []
        n_sites_final = []
        converged_early = False
        for i_motif in range(self.n_motifs):
            print('\nFinding motif %i of %i' % (i_motif+1, self.n_motifs))
            X_seq = X
            X_neg_seq = X_neg
            N = len(X)
            if X_neg is not None:
                top_words = initialize.find_enriched_gapped_kmers(X, X_neg, self.half_length, 0,
                                                                  self.motif_width - 2 * self.half_length,
                                                                  self.alpha, self.revcomp, self.n_seeds)
            X = sequences.encode(X, self.alpha)#need to change this to X_seq
            X_seqs_onehot = X # Need to use one hot coded positive sequences later
            if X_neg is not None:
                # Need to use one hot coded negative sequences later
                X_neg_seqs_onehot = sequences.encode(X_neg, self.alpha)
            # Extract valid one-hot subsequences
            X = sequences.get_onehot_subsequences(X, self.motif_width)
            M, L, W = X.shape
            if self.revcomp:
                M *= 2
            # Compute motif fractions seeds
            min_sites = self.min_sites
            min_frac = min_sites / M
            if self.max_sites is None:
                max_sites = N # Expect at most one motif occurrence per sequence by default
            else:
                max_sites = self.max_sites
            max_frac = max_sites / M
            fracs_seeds = np.geomspace(min_sites, max_sites, 5) / M
            n_uniq_fracs_seeds = len(fracs_seeds)
            fracs_seeds = np.repeat(fracs_seeds, self.n_seeds)
            fracs_seeds = torch.from_numpy(fracs_seeds.astype(np.float32))
            # Compute background frequencies
            letter_frequency = X.sum(axis=(0,2))
            if self.revcomp:  # If reverse complements considered, complement letter frequencies set to same value
                letter_frequency[[0, 3]] = letter_frequency[0] + letter_frequency[3]
                letter_frequency[[1, 2]] = letter_frequency[1] + letter_frequency[2]
                X = np.concatenate((X, X[:, ::-1, ::-1]), axis=0)
            bg_probs = 1.0 * letter_frequency / letter_frequency.sum()
            ppms_bg_seeds = bg_probs.reshape([1, L, 1]).repeat(
                                self.n_seeds * n_uniq_fracs_seeds, axis=0).astype(np.float32)
            ppms_bg_seeds = torch.from_numpy(ppms_bg_seeds)
            # Initialize PPMs
            large_prob = 0.9
            small_prob = (1 - large_prob) / (L - 1)
            if X_neg is not None:
                ppms_seeds = sequences.encode(top_words, self.alpha)
                ppms_seeds = sequences.pad_onehot_sequences(ppms_seeds, W).astype(np.float32) * large_prob
                for ppm in ppms_seeds:
                    ppm[:, ppm.sum(axis=0)==0] = bg_probs.reshape((L, 1))
                ppms_seeds[ppms_seeds == 0] = small_prob
            else:
                ppms_seeds = X[0:self.n_seeds].astype(np.float32) * large_prob
                ppms_seeds[ppms_seeds == 0] = small_prob
            ppms_seeds = np.tile(ppms_seeds, (n_uniq_fracs_seeds, 1, 1))
            ppms_seeds_original = ppms_seeds.copy()
            ppms_seeds = torch.from_numpy(ppms_seeds)
            # If using cuda, convert the three parameter tensors to cuda format
            if self.cuda:
                ppms_bg_seeds = ppms_bg_seeds.cuda()
                ppms_seeds = ppms_seeds.cuda()
                fracs_seeds = fracs_seeds.cuda()
            ppms_bg_seeds = ppms_bg_seeds.expand(len(ppms_bg_seeds), L, W)

            # Perform one On-line and one batch EM pass
            ppms_seeds, ppms_bg_seeds, fracs_seeds = \
                self._online_em(X, ppms_seeds, ppms_bg_seeds, fracs_seeds, 1)
            ppms, ppms_bg, fracs = \
                self._batch_em(X, ppms_seeds, ppms_bg_seeds, fracs_seeds, 1)
            log_likelihoods = self._compute_log_likelihood(X, ppms, ppms_bg, fracs)
            # Filter away all invalid parameter sets
            # Removed the right-most filter since it was causing issues for some people
            bool_mask = (log_likelihoods != np.inf)  #& (fracs > min_frac) & (fracs < max_frac)
            indices = torch.arange(0, len(bool_mask), 1).long()
            if self.cuda:
                indices = indices.cuda()
            if len(indices) == 0:
                converged_early = True
                break
            indices = indices[bool_mask]
            log_likelihoods = log_likelihoods[indices]
            ppms = ppms[indices]
            ppms_bg = ppms_bg[indices]
            fracs = fracs[indices]
            ppms_seeds = ppms_seeds[indices]
            # Select seed that yields highest log likelihood after one online and one batch EM passes
            max_log_likelihoods, max_log_likelihoods_index = log_likelihoods.max(dim=0)
            max_log_likelihoods_index = max_log_likelihoods_index.item()  # Replaced [0] w/ .item() for PyTorch >= 0.4
            word_seed_best = sequences.decode(
                [ppms_seeds_original[max_log_likelihoods_index].round().astype(np.uint8)], self.alpha)[0]
            print('Using seed originating from word: %s' % (word_seed_best))
            ppm_best = ppms[[max_log_likelihoods_index]]
            ppm_bg_best = ppms_bg[[max_log_likelihoods_index]]
            frac_best = fracs[[max_log_likelihoods_index]]
            # Refine the best seed with batch EM passes
            ppm_best, ppm_bg_best, frac_best = \
                self._batch_em(X, ppm_best, ppm_bg_best, frac_best, self.maxiter)
            if np.isnan(ppm_best[0].cpu().numpy()).any():
                converged_early = True
                break
            ppms_final.append(ppm_best[0].cpu().numpy())
            ppms_bg_final.append(ppm_bg_best[0].cpu().numpy())
            fracs_final.append(frac_best[0])
            n_sites = M * fracs_final[-1].cpu()
            if np.isnan(n_sites):
                n_sites = 0
            else:
                n_sites = int(n_sites)
            n_sites_final.append(n_sites)
            if self.erasewhole:
                print('\nRemoving sequences containing at least one motif occurrence')
                X = self._erase_seqs_containing_motifs(X_seqs_onehot, ppms_final[-1], ppms_bg_final[-1],
                                                       fracs_final[-1])
                if X_neg is not None:
                    X_neg = self._erase_seqs_containing_motifs(X_neg_seqs_onehot, ppms_final[-1], ppms_bg_final[-1],
                                                               fracs_final[-1])
            else:
                print('\nRemoving individual occurrences of motif occurrences')
                X = self._erase_motif_occurrences(X_seqs_onehot, ppms_final[-1], ppms_bg_final[-1], fracs_final[-1])
                if X_neg is not None:
                    X_neg = self._erase_motif_occurrences(X_neg_seqs_onehot, ppms_final[-1], ppms_bg_final[-1],
                                                          fracs_final[-1])
            X_seq = X
            X_neg_seq = X_neg
        if converged_early:
            print('\n\nYou asked to find %i motifs, but YAMDA found only %i motifs' % (self.n_motifs, len(ppms_final)))
        self.ppms_ = ppms_final
        self.ppms_bg_ = ppms_bg_final
        self.fracs_ = fracs_final
        self.n_sites_ = n_sites_final
        return X_seq, X_neg_seq

    def _batch_em(self, X, ppms, ppms_bg, fracs, epochs):
        M, L, W = X.shape
        n_filters = len(ppms)
        m_log_ratios = nn.Conv1d(L, n_filters, W, stride=W, bias=False)
        fracs = fracs.view((1, n_filters, 1))
        pfms = torch.zeros((n_filters, L, W))
        pfms_bg = torch.zeros((n_filters, L, W))
        counts = torch.zeros((n_filters, 1))
        if self.cuda:
            m_log_ratios.cuda()
            pfms = pfms.cuda()
            pfms_bg = pfms_bg.cuda()
            counts = counts.cuda()
        converged = False
        pbar_epoch = trange(0, epochs, 1, desc='Batch EM')
        for i in pbar_epoch:
            if converged:
                continue
            old_ppms = ppms
            # E-step, compute membership weights and letter frequencies
            pfms.zero_()
            pfms_bg.zero_()
            counts.zero_()
            m_log_ratios.weight.data = torch.log(ppms) - torch.log(ppms_bg)
            fracs_ratio = fracs / (1 - fracs)
            for j in trange(0, M, self.batch_size, desc='Pass %i/%i' % (i + 1, epochs)):
                batch = X[j:j + self.batch_size]
                x = Variable(torch.from_numpy(batch).float())
                if self.cuda:
                    x = x.cuda()
                log_ratios = m_log_ratios(x).data
                ratios = torch.exp(log_ratios)
                c = self.fudge * fracs_ratio * ratios
                state_probs = c / (1 + c)
                counts.add_(state_probs.sum(dim=0))
                batch_motif_matrix_counts = (state_probs.unsqueeze(-1) *
                                             x.data.unsqueeze(1)).sum(dim=0)
                pfms.add_(batch_motif_matrix_counts)
                pfms_bg.add_(x.data.sum(dim=0).unsqueeze(0) - batch_motif_matrix_counts)
            # M-step, update parameters
            fracs = (counts / M).unsqueeze(0)
            ppms = pfms / counts.unsqueeze(2)
            ppms_bg = (pfms_bg.sum(dim=-1) /
                       (W * (M - counts))).unsqueeze(2).expand(n_filters, L, W)
            ppms_diff_norm = (ppms - old_ppms).view(n_filters, -1).norm(p=2, dim=1)
            max_ppms_diff_norm = ppms_diff_norm.max()
            if max_ppms_diff_norm < self.tolerance:
                pbar_epoch.set_description('Batch EM - convergence reached after %i epochs' % (i+1))
                converged = True
        fracs = fracs.view(-1)
        return ppms, ppms_bg, fracs

    def _online_em(self, X, ppms, ppms_bg, fracs, epochs):
        M, L, W = X.shape
        n_filters = len(ppms)
        m_log_ratios = nn.Conv1d(L, n_filters, W, stride=W, bias=False)
        fracs = fracs.view((1, n_filters, 1))
        # On-line EM specific-parameters
        gamma_0 = 0.5
        alpha = 0.85
        s_0 = fracs.clone()[0].unsqueeze(-1)
        s_1 = s_0 * ppms
        s_1_bg = (1 - s_0) * ppms_bg
        k = 0
        indices = np.random.permutation(M)
        if self.cuda:
            m_log_ratios.cuda()
            s_0 = s_0.cuda()
            s_1 = s_1.cuda()
            s_1_bg = s_1_bg.cuda()
        pbar_epoch = trange(0, epochs, 1, desc='On-line EM')
        converged = False
        for i in pbar_epoch:
            if converged:
                continue
            old_ppms = ppms
            for j in trange(0, M, self.batch_size, desc='Pass %i/%i' % (i + 1, epochs)):
                k += 1
                m_log_ratios.weight.data = torch.log(ppms) - torch.log(ppms_bg)
                fracs_ratio = fracs / (1 - fracs)
                # E-step, compute membership weights and letter frequencies for a batch
                batch = X[indices[j:j + self.batch_size]]
                actual_batch_size = len(batch)
                gamma = 1.0 * actual_batch_size / self.batch_size * gamma_0 / (k ** alpha)
                x = Variable(torch.from_numpy(batch).float())
                if self.cuda:
                    x = x.cuda()
                log_ratios = m_log_ratios(x).data
                ratios = torch.exp(log_ratios)
                c = self.fudge * fracs_ratio * ratios
                state_probs = c / (1 + c)
                s_0_temp = state_probs.mean(dim=0).unsqueeze(-1)
                s_1_temp = (state_probs.unsqueeze(-1) *
                            x.data.unsqueeze(1)).mean(dim=0)
                s_1_bg_temp = x.data.mean(dim=0).unsqueeze(0) - s_1_temp
                # M-step, update parameters based on batch
                s_0.add_(gamma * (s_0_temp - s_0))
                s_1.add_(gamma * (s_1_temp - s_1))
                s_1_bg.add_(gamma * (s_1_bg_temp - s_1_bg))
                fracs = s_0.view((1, n_filters, 1))
                ppms = s_1 / s_0
                ppms_bg = (s_1_bg / (1 - s_0)).mean(-1, keepdim=True).expand((n_filters, L, W))
            ppms_diff_norm = (ppms - old_ppms).view(n_filters, -1).norm(p=2, dim=1)
            max_ppms_diff_norm = ppms_diff_norm.max()
            if max_ppms_diff_norm < self.tolerance:
                pbar_epoch.set_description('On-line EM - convergence reached')
                converged = True
        fracs = fracs.view(-1)
        return ppms, ppms_bg, fracs

    def _compute_log_likelihood(self, X, ppms, ppms_bg, fracs):
        M, L, W = X.shape
        n_filters = len(ppms)
        m_log_ppms_bg = nn.Conv1d(L, n_filters, W, bias=False)
        m_log_ppms_bg.weight.data = torch.log(ppms_bg)
        m_log_ratios = nn.Conv1d(L, n_filters, W, bias=False)
        m_log_ratios.weight.data = torch.log(ppms) - torch.log(ppms_bg)
        fracs = fracs.view((1, n_filters, 1))
        log_likelihoods = torch.zeros(n_filters)
        fracs_ratio = fracs / (1 - fracs)
        log_fracs_bg = torch.log(1 - fracs)
        if self.cuda:
            m_log_ppms_bg.cuda()
            log_likelihoods = log_likelihoods.cuda()
            m_log_ratios.cuda()
        for j in trange(0, M, self.batch_size, desc='Computing log likelihood'):
            batch = X[j:j + self.batch_size]
            x = Variable(torch.from_numpy(batch).float())
            if self.cuda:
                x = x.cuda()
            ppms_bg_logprob = m_log_ppms_bg(x).data
            log_ratios = m_log_ratios(x).data
            ratios = torch.exp(log_ratios)
            # Added back self.fudge here, since this is the quantity that EM is technically optimizing
            log_likelihoods.add_((log_fracs_bg + ppms_bg_logprob +
                                  torch.log(1 + self.fudge * fracs_ratio * ratios)).sum(dim=0).view(-1))
        return log_likelihoods

    def _erase_motif_occurrences(self, seqs_onehot, ppm, ppm_bg, frac):
        t = np.log((1 - frac) / frac)  # Threshold
        spec = np.log(ppm) - np.log(ppm_bg)  # spec matrix
        spec_revcomp = spec[::-1, ::-1]
        L, W = ppm.shape
        for i in range(0, len(seqs_onehot), 1):
            s = seqs_onehot[i]  # grab the one hot coded sequence
            seqlen = s.shape[1]
            if seqlen < W: # leave short sequences alone
                continue
            indices = np.arange(seqlen - W + 1)
            conv_signal = signal.convolve2d(spec, s, 'valid')[0]
            seq_motif_sites = indices[conv_signal > t]
            if self.revcomp:
                conv_signal_revcomp = signal.convolve2d(spec_revcomp, s, 'valid')[0]
                seq_motif_sites_revcomp = indices[conv_signal_revcomp > t]
                seq_motif_sites = np.concatenate((seq_motif_sites, seq_motif_sites_revcomp))
            for motif_site in seq_motif_sites:
                s[:, motif_site:motif_site+W] = 0
        seqs = sequences.decode(seqs_onehot, self.alpha)
        return seqs

    def _erase_seqs_containing_motifs(self, seqs_onehot, ppm, ppm_bg, frac):
        t = np.log((1 - frac) / frac)  # Threshold
        spec = np.log(ppm) - np.log(ppm_bg)  # spec matrix
        spec_revcomp = spec[::-1, ::-1]
        L, W = ppm.shape
        seqs_onehot_filtered = []
        for i in range(0, len(seqs_onehot), 1):
            s = seqs_onehot[i]  # grab the one hot coded sequence
            if s.shape[1] < W: # leave short sequences alone
                seqs_onehot_filtered.append(s)
                continue
            conv_signal = signal.convolve2d(spec, s, 'valid')[0]
            s_has_motif = any(conv_signal > t)
            if self.revcomp:
                conv_signal_revcomp = signal.convolve2d(spec_revcomp, s, 'valid')[0]
                s_has_motif = s_has_motif or any(conv_signal_revcomp > t)
            if not s_has_motif:
                seqs_onehot_filtered.append(s)
        seqs = sequences.decode(seqs_onehot_filtered, self.alpha)
        return seqs

