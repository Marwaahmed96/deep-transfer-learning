from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
#from torch.utils import model_zoo
from config.settings import *
from utils.data_preprocess import *
import dataset_loader as dl
from torch.utils.tensorboard import SummaryWriter
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
writer = SummaryWriter('runs')

# Training settings
st = Settings()
options = st.get_options()

batch_size = options['batch_size']
epochs = options['max_epochs']
lr = [0.001, 0.01]
momentum = 0.9
no_cuda =False
seed = options['seed']
log_interval = 10
l2_decay = 5e-4
source_path = options['train_folder']
source_name = 'miccai'
cuda = not no_cuda and torch.cuda.is_available()

# resize images in path
#resize_images(options)

# generate csv file
df = generate_csv(options)

# split data to train, validate folds
#split_folds(options['train_csv_path'], options['seed'], options['k_fold'])

# list scan
fold = 0
# fold train data
df = pd.read_csv(options['train_csv_path'])
# select training scans
train_files = df.loc[df['fold'] != fold, ['patient_id','study']].values
valid_files = df.loc[df['fold'] == fold, ['patient_id', 'study']].values
train_scan_list = [f[0]+f[1] for f in train_files]
valid_scan_list = [f[0]+f[1] for f in valid_files]

train_scan_list.sort()
valid_scan_list.sort()

train_x_data = {f: {m: os.path.join(options['train_folder'], f, options['tmp_folder'], n)
                    for m, n in zip(options['modalities'], options['preprocess_x_names'])}
                for f in train_scan_list}
train_y_data = {f: os.path.join(options['train_folder'], f, options['tmp_folder'],
                                options['preprocess_y_names'][0])
                for f in train_scan_list}

valid_x_data = {f: {m: os.path.join(options['train_folder'], f, options['tmp_folder'], n)
                    for m, n in zip(options['modalities'], options['preprocess_x_names'])}
                for f in valid_scan_list}
valid_y_data = {f: os.path.join(options['train_folder'], f, options['tmp_folder'],
                                options['preprocess_y_names'][0])
                for f in valid_scan_list}


torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source_train_loader = dl.load_training(options, train_x_data, train_y_data)
source_valid_loader = dl.load_training(options, train_x_data, train_y_data)

#source_test_loader = data_loader.load_testing('', source_path, batch_size, kwargs)

len_source_train_dataset = len(source_train_loader.dataset)
len_source_valid_dataset = len(source_valid_loader.dataset)
len_source_train_loader = len(source_train_loader)
len_source_valid_loader = len(source_valid_loader)


def train(epoch, model, optimizer):

    optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)

    model.train()

    iter_source_train = iter(source_train_loader)
    num_iter_train = len_source_train_loader
    for i in range(1, num_iter_train):
        data_source_train, label_source_train = iter_source_train.next()
        if cuda:
            data_source_train, label_source_train = data_source_train.cuda(), label_source_train.cuda()
        data_source_train, label_source_train = Variable(data_source_train), Variable(label_source_train)

        optimizer.zero_grad()
        label_source_train_pred = model(data_source_train)
        loss = F.nll_loss(F.log_softmax( label_source_train_pred, dim=1), torch.tensor(label_source_train,dtype=torch.long))
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}'.format(
                epoch, i * len(data_source_train), len_source_train_dataset,
                100. * i / len_source_train_loader, loss.data[0], loss.data[0]))

def validate(model):
    model.eval()
    test_loss = 0
    correct = 0

    for data in source_valid_loader:
        if cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        s_output = model(data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), size_average=False).data[0] # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(data.view_as(pred)).cpu().sum()

    test_loss /= len_source_valid_dataset
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        source_name, test_loss, correct, len_source_valid_dataset,
        100. * correct / len_source_valid_dataset))
    return correct


if __name__ == '__main__':
    model = models.DANNet_source(num_classes=2)
    writer.add_graph(model, torch.rand(size=(128,2,16,16,16)))
    writer.flush()
    writer.close()
    sys.exit()
    correct = 0
    print(model)
    if cuda:
        model.cuda()

    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': lr[1]},
        ], lr=lr[0], momentum=momentum, weight_decay=l2_decay)

    for epoch in range(1, epochs + 1):
        train(epoch, model, optimizer)
        t_correct = validate(model)
        if t_correct > correct:
            correct = t_correct
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
              source_name, correct, 100. * correct / len_source_valid_dataset))
