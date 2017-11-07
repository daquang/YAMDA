from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Dense, Dropout, Activation, Flatten, Layer, merge, Input
from keras.layers.pooling import GlobalMaxPooling1D
import numpy as np

from . import sequences

class SeqDiscrim():
    def __init__(self, n_seeds, motif_width, backend="keras"):
        self.n_seeds = n_seeds
        self.motif_width = motif_width
        self.backend = backend

    def fit(X, X2):
        max_seq_len = 0
        for x in X:
            max_seq_len = max(max_seq_len, x.shape[1])
        for x in X2:
            max_seq_len = max(max_seq_len, x.shape[1])
        X = sequences.pad_sequences(X, max_seq_len)
        X2 = sequences.pad_sequences(X2, max_seq_len)
        Y = np.zeros(len(X) + len(X2), dtype=np.uint8)
        Y[:len(X)] = 1
        X = np.vstack([X, X2])
        N, L, _ = X.shape
        if self.backend == "keras":
            self.model = _make_model(max_seq_len, L)

    def _get_output(input_layer, hidden_layers):
        output = input_layer
        for hidden_layer in hidden_layers:
            output = hidden_layer(output)
        return output

    def _make_model(max_seq_len, L):
        rc_layer = Lambda(lambda x: x[:,::-1,::-1])
        hidden_layers = [
            Convolution1D(input_dim = L,
                          nb_filter = self.n_seeds,
                          filter_length = self.motif_width, 
                          border_mode = 'valid', 
                          activation = 'relu',
                          subsample_length = 1),
            GlobalMaxPooling1D(),
            Flatten(),
            Dense(1, activation = 'sigmoid')
        ]
        forward_input = Input(shape=(max_seq_len, L,))
        reverse_input = rc_layer(forward_input)
        output = self.
        model = Model(input=[forward_input], output=output)
        return model

    def transform(X):
        return

