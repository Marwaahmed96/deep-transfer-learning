from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from utils.data_load import *
from config.settings import *


def load_target_voxels(train_x_data, options):
    # get_scan names and number of modalities used
    scans = list(train_x_data.keys())
    modalities = train_x_data[scans[0]].keys()
    flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
    # load images and normalize their intensities
    images = [load_nii(image_name).get_data() for image_name in flair_scans]
    images_norm = [normalize_data(im) for im in images]
    # select voxels with intensity higher than threshold
    selected_voxels = [image > options['min_th'] for image in images_norm]
    data = []
    random_state=42
    datatype=np.float32
    patch_size = options['patch_size']

    for m in modalities:
        x_data = [train_x_data[s][m] for s in scans]
        images = [load_nii(name).get_data() for name in x_data]
        images_norm = [normalize_data(im) for im in images]
        # Get all the x,y,z coordinates for each image
        centers = [get_mask_voxels(mask) for mask in selected_voxels]

        x_patches = [np.array(get_patches(image, centers, patch_size))
                         for image, centers in zip(images_norm, centers)]

        data.append(np.concatenate(x_patches))
    X = np.stack(data, axis=1)
    return X


class GenerateDataset(Dataset):
    # dictionary of x_data files names, y_data files names
    def __init__(self, options, x_data, y_data, transform, model=None):
        print("> CNN: loading training data")
        if y_data is None:
            print('y_data is none')
            X = load_target_voxels(x_data, options)
            Y = None

        else:
            X, Y, sel_voxels = load_training_data(x_data, y_data, options, model=model)

        if y_data is None:
            print('> CNN: train_x ', X.shape, 'train_y ')
        else:
            print('> CNN: train_x ', X.shape, 'train_y ', Y.shape)

        self.batch_size = options['batch_size']
        self.transform = transform
        self.x_train = torch.from_numpy(X)
        self.y_train = None if Y is None else torch.from_numpy(Y)
        #self.x_train = self.transform(X)
        #self.y_train = self.transform(Y)
        self.n_samples = self.x_train.shape[0]

    def __getitem__(self, index):
        if self.y_train is None:
            return self.x_train[index, :]
        else:
            return self.x_train[index, :], self.y_train[index]

    def __len__(self):
        return self.n_samples


def load_training(options, x_data, y_data, model=None):
    batch_size = options['batch_size']
    transform = transforms.Compose([transforms.ToTensor()])

    data = GenerateDataset(options=options, x_data=x_data, y_data=y_data, transform=transform, model=model)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader


def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader
