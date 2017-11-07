import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.conv2_drop(x)
        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return F.log_softmax(x)

class SeqDiscrim():
    def __init__(self):

    def fit(X, X2):
        max_seq_len = 0
        for x in X:
            max_seq_len = max(max_seq_len, x.shape[1])
        for x in X2:
            max_seq_len = max(max_seq_len, x.shape[1])
        
        cum_loss = 0
        model.train()
        pbar = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            cum_loss += loss.data[0]
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                pbar.set_description('Epoch %i/%i - loss: %0.4f' % (epoch, args.epochs, cum_loss / (batch_idx + 1)))


    def transform(X):

