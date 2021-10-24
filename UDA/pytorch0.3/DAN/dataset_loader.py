from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from utils.data_load import *
from config.settings import *


class GenerateDataset(Dataset):
    # dictionary of x_data files names, y_data files names
    def __init__(self, options, x_data, y_data, transform):
        print("> CNN: loading training data for first model")
        X, Y, sel_voxels = load_training_data(x_data, y_data, options)
        print('> CNN: train_x ', X.shape, 'train_y ', Y.shape)
        self.batch_size = options['batch_size']
        self.transform = transform
        self.x_train = torch.from_numpy(X)
        self.y_train = torch.from_numpy(Y)
        #self.x_train = self.transform(X)
        #self.y_train = self.transform(Y)
        self.n_samples = self.x_train.shape[0]

    def __getitem__(self, index):
        return self.x_train[index, :], self.y_train[index]

    def __len__(self):
        return self.n_samples


def load_training(options, x_data, y_data):
    batch_size = options['batch_size']
    transform = transforms.Compose([transforms.ToTensor()])

    data = GenerateDataset(options=options, x_data=x_data, y_data=y_data, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader


def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader
