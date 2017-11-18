"""
from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Dense, Dropout, Activation, Flatten, Layer, merge, Input, Convolution1D, MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import trange
from sklearn.utils import shuffle

from . import sequences


class Net(nn.Module):
    def __init__(self, L, n_motifs, motif_width, revcomp=False):
        super(Net, self).__init__()
        self.revcomp = revcomp
        self.conv = nn.Conv1d(L, n_motifs, motif_width)
        self.dense = nn.Linear(n_motifs, 1)
        # Initialization
        nn.init.kaiming_normal(self.conv.weight.data)
        self.conv.bias.data.zero_()
        nn.init.xavier_normal(self.dense.weight.data)
        self.dense.bias.data.zero_()

    def forward(self, x):
        x_fwd = x
        x_fwd = self.conv(x_fwd)
        x_fwd = x_fwd.max(dim=2, keepdim=False)[0]
        x_fwd = F.relu(x_fwd)
        x_fwd = self.dense(x_fwd)
        output = F.sigmoid(x_fwd)
        return output
    
    @staticmethod
    def _flip(x, dim):
        dim = x.dim() + dim if dim < 0 else dim
        return x[tuple(slice(None, None) if i != dim
                 else torch.arange(x.size(i)-1, -1, -1).long()
                 for i in range(x.dim()))]


class SeqDiscrim():
    def __init__(self, n_motifs, motif_width, batch_size, alpha, cuda):
        self.n_motifs = n_motifs
        self.motif_width = motif_width
        self.batch_size = batch_size
        self.alpha = alpha
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
        X_pos = sequences.pad_onehot_sequences(X_pos, max_seq_len)
        X_neg = sequences.pad_onehot_sequences(X_neg, max_seq_len)
        X = np.vstack([X_pos, X_neg])
        N, L, _ = X.shape
        Y = np.zeros((N, 1), dtype=np.uint8)
        Y[:len(X_pos), 0] = 1
        self.model = Net(L, self.n_motifs, self.motif_width)
        if self.cuda:
            self.model.cuda()
        self._train(X, Y)
        """
        if True:
            self.model = self._make_model(max_seq_len, L)
            self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
            self.model.summary()
            X = X.transpose((0,2,1))
        self.model.fit(X,Y,batch_size=100,epochs=100,verbose=1,shuffle=True)
        """
    
    def _train(self, X, Y, epochs=100):
        self.model.train()
        N = len(X)
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0)
        #optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            cum_loss = 0
            cum_acc = 0
            pbar = trange(0, N, self.batch_size)
            k = 0
            for j in pbar:
                x = X[j:j+self.batch_size]
                y = Y[j:j+self.batch_size]
                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).float()
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()
                x, y = Variable(x), Variable(y)
                optimizer.zero_grad()
                output = self.model(x)
                predicts = output.round()
                acc = (predicts.eq(y).sum()).data[0] / len(y)
                cum_acc += acc
                loss = F.binary_cross_entropy(output, y)
                cum_loss += loss.data[0]
                loss.backward()
                optimizer.step()
                if k % 5 == 0:
                    pbar.set_description('Epoch %i/%i - loss: %0.4f acc: %0.4f' % (i+1, epochs, cum_loss / (k+1), cum_acc / (k+1)))
                k += 1
                
        
    """
    @staticmethod
    def _get_output(input_layer, hidden_layers):
        output = input_layer
        for hidden_layer in hidden_layers:
            output = hidden_layer(output)
        return output
    """

    """
    def _make_model(self, max_seq_len, L):
        conv_layer = torch.nn.Conv1d(L, self.n_motifs, self.motif_width)
        n_scanpoints = max_seq_len - self.motif_width + 1
        dense_layer = torch.nn.Linear(n_scanpoints, 1)
        model_fwd = torch.nn.Sequential(
          conv_layer,
          torch.nn.ReLU(),
          torch.nn.MaxPool1d(n_scanpoints),
          
          torch.nn.Linear(H, D_out),
        )
        rc_layer = Lambda(lambda x: x[:,::-1,::-1])
        hidden_layers = [
            Convolution1D(filters=self.n_motifs,
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
        return model_fwd
        """

    def transform(self, X):
        return

