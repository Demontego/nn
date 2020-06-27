import os
import torch
import sys
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import MSELoss, Sequential, Linear, Sigmoid, Tanh
from torch.autograd import Variable
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from torch import nn
landmarks_frame = pd.read_csv('dataset/train/face_landmarks.csv')
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix().astype('float')
        landmarks = landmarks.reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        return {'image':  torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
class ToTensor2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        return {'image':  torch.from_numpy(image+np.random.randn(96*96).reshape(96, 96)*0.32),
                'landmarks': torch.from_numpy(landmarks)}
    
train_dataset = FaceLandmarksDataset(csv_file='dataset/train/face_landmarks.csv',
                                     root_dir='dataset/train',
                                     transform=ToTensor2()
                                     )

test_dataset = FaceLandmarksDataset(csv_file='dataset/test/face_landmarks.csv',
                                     root_dir='dataset/test',
                                     transform=ToTensor()
                                     )
train_dataloader = DataLoader(train_dataset, batch_size=64,
                        shuffle=True, num_workers=60)

test_dataloader = DataLoader(test_dataset, batch_size=64,
                        shuffle=True,num_workers=60)

dtype=torch.FloatTensor

def train(network, epochs, learning_rate, loss=MSELoss(), optim=torch.optim.Adam):
    train_loss_epochs = []
    test_loss_epochs = []
    optimizer = optim(network.parameters(), lr=learning_rate)
    try:
        for epoch in range(epochs):
            losses = []
            accuracies = []
            for sample in train_dataloader:
                X = sample['image']
                X = X.view(X.shape[0],1,96,96).type(dtype)
                #print(X.shape)
                y = sample['landmarks']
                y = y.view(y.shape[0], -1).type(dtype)
                
                prediction = network(X)
                loss_batch = loss(prediction, y)
                losses.append(loss_batch.item())
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
  
            train_loss_epochs.append(np.mean(losses))
            losses = []    
            for sample in test_dataloader:
                X = sample['image']
                X = X.view(X.shape[0],1,96,96).type(dtype)
                #X = X.view(X.shape[0], -1).type(dtype)
                y = sample['landmarks']
                y = y.view(y.shape[0], -1).type(dtype)
                
                prediction = network(X)
                loss_batch = loss(prediction, y)
                losses.append(loss_batch.item())
                
            test_loss_epochs.append(np.mean(losses))
            #sys.stdout.write('\rEpoch {0}... (Train/Test) MSE: {1:.3f}/{2:.3f}'.format(
              #          epoch, train_loss_epochs[-1], test_loss_epochs[-1]))
            print(epoch, train_loss_epochs[-1], test_loss_epochs[-1])
    except KeyboardInterrupt:
        pass
def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_normal(layer.weight)
channels = 1
class ConvClassifier(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(channels, 8, (4,4), padding=(2,2)), nn.ReLU(),
                                        nn.MaxPool2d(2),
                                         nn.Conv2d(8, 16, (4,4), padding=(2,2)), nn.ReLU(),
                                         nn.MaxPool2d(2),
                                        nn.Conv2d(16, 32, (4,4), padding=(2,2)),nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Conv2d(32, 64, (4,4), padding=1),nn.ReLU(),
                                        nn.MaxPool2d(2))
        self.linear_layers = nn.Sequential(nn.Linear(40*40, 400),
                                           nn.BatchNorm1d(400),
                                           nn.ReLU(),nn.Linear(400, 200),nn.ReLU(),
                                           nn.BatchNorm1d(200),
                                          nn.Linear(200, 2*68))
        self.linear_layers.apply(init_weights)
        

    def forward(self, x):
        #print(x.shape)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        #print(x.shape)
        return x
network = ConvClassifier(image_size=96)
train(network, 750, 0.0001)
