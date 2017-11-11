from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Dense, Dropout, Activation, Flatten, Layer, merge, Input, Convolution1D, MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
import numpy as np

from . import sequences


class SeqDiscrim():
    def __init__(self, n_seeds=16, motif_width=25, backend="keras"):
        self.n_seeds = n_seeds
        self.motif_width = motif_width
        self.backend = backend
        self.model = None

    def fit(self, X, X2, n_seeds, motif_width, min_sites,
                 batch_size, cuda=True, init='subsequences'):
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
        if self.backend == 'keras':
            self.model = self._make_model(max_seq_len, L)
            self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
            self.model.summary()
            X = X.transpose((0,2,1))
        self.model.fit(X,Y,batch_size=100,epochs=100,verbose=1,shuffle=True)

    @staticmethod
    def _get_output(input_layer, hidden_layers):
        output = input_layer
        for hidden_layer in hidden_layers:
            output = hidden_layer(output)
        return output

    def _make_model(self, max_seq_len, L):
        rc_layer = Lambda(lambda x: x[:,::-1,::-1])
        hidden_layers = [
            Convolution1D(filters=self.n_seeds,
                          kernel_size=self.motif_width,
                          padding='valid',
                          activation='relu'),
            GlobalMaxPooling1D(),
            Dense(1, activation='sigmoid')
        ]
        forward_input = Input(shape=(max_seq_len, L,))
        reverse_input = rc_layer(forward_input)
        output = self._get_output(forward_input, hidden_layers)
        model = Model(input=[forward_input], output=output)
        return model

    def transform(self, X):
        return

