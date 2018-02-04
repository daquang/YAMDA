from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPool1D, Input, Dense, Lambda, Maximum
from keras import backend as K
from tqdm import trange
import numpy as np
from sklearn.model_selection import train_test_split

from . import sequences


class SeqDiscrim():
    def __init__(self, n_motifs, motif_width, test_size, batch_size, epochs, alpha, revcomp, cuda):
        self.n_motifs = n_motifs
        self.motif_width = motif_width
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.alpha = alpha
        self.revcomp = revcomp
        self.cuda = cuda
        self.model = None

    def fit(self, X_pos, X_neg):
        max_seq_len = 0
        for x in X_pos:
            max_seq_len = max(max_seq_len, len(x))
        for x in X_neg:
            max_seq_len = max(max_seq_len, len(x))
        X_pos = sequences.encode(X_pos, self.alpha)
        X_neg = sequences.encode(X_neg, self.alpha)
        X_pos = sequences.pad_onehot_sequences(X_pos, max_seq_len, center=False)
        X_neg = sequences.pad_onehot_sequences(X_neg, max_seq_len, center=False)
        X = np.vstack([X_pos, X_neg])
        N, A, L = X.shape
        y = np.zeros((N, 1), dtype=np.uint8)
        y[:len(X_pos), 0] = 1
        if self.test_size is None:
            X_train = X
            y_train = y
            X_test = X
            y_test = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                stratify=y,
                                                                test_size=self.test_size)
        self.conv_layer, self.model = self._make_model(A, L)
        self._train(X_train, y_train)
        ppms, nsites = self._filters_to_ppms(X_pos)
        return ppms, nsites

    def _make_model(self, A, L):
        input_layer = Input((L, A))
        conv_layer = Conv1D(filters=self.n_motifs,
                            kernel_size=self.motif_width,
                            strides=1,
                            padding='valid',
                            activation='relu')
        pool_layer = GlobalMaxPool1D()
        sigmoid_layer = Dense(1,
                              activation='sigmoid')

        rev_comp_layer = Lambda(lambda x: x[:, ::-1, ::-1])

        output_fwd = conv_layer(input_layer)
        output_fwd = pool_layer(output_fwd)
        output_fwd = sigmoid_layer(output_fwd)

        if self.revcomp:
            output_rev = conv_layer(rev_comp_layer(input_layer))
            output_rev = pool_layer(output_rev)
            output_rev = sigmoid_layer(output_rev)

            output = Maximum()([output_fwd, output_rev])
        else:
            output = output_fwd

        model = Model([input_layer], [output])

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return conv_layer, model

    def _train(self, X, y):
        X = X.transpose(0, 2, 1)
        self.model.fit(X, y,
                  batch_size=100,
                  epochs=self.epochs,
                  shuffle=True)

    def _filters_to_ppms(self, X):
        N, A, L = X.shape
        X = X.transpose(0, 2, 1)
        pfms = np.zeros((self.n_motifs, self.motif_width, A))
        nsites = np.zeros(self.n_motifs)

        conv_output = self.conv_layer.get_output_at(0)
        f = K.function([self.model.input], [K.argmax(conv_output, axis=1), K.max(conv_output, axis=1)])
        pbar = trange(0, N, self.batch_size)
        for i in pbar:
            x_batch = X[i:i + self.batch_size]
            max_acts_pos_batch, max_acts_batch = f([x_batch])
            max_acts_batch_bool = max_acts_batch > 0
            for m in range(self.n_motifs):
                for n in range(len(x_batch)):
                    if max_acts_batch_bool[n, m]:
                        nsites[m] += 1
                        pfms[m] += x_batch[n, max_acts_pos_batch[n, m]:max_acts_pos_batch[n, m] + self.motif_width, :]
        ppms = pfms / pfms.sum(axis=2, keepdims=True)
        ppms = ppms.transpose(0, 2, 1)
        return ppms, nsites

    def transform(self, X):
        return
