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
from torch.utils.tensorboard import SummaryWriter
from config.Domain_Adapt_settings import *
import pandas as pd
import dataset_loader as dl
from pathlib import Path


def load_pretrain(model):
    FILE = os.path.join(options['weight_paths'], options['source_experiment'], options['pre_trained_model'])
    print(FILE)
    model = torch.load(FILE)
    return model


def train(epoch, model, optimizer):

    optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)

    model.train()
    train_loss=0
    correct = 0

    iter_source = iter(source_train_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_train_loader
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target = iter_target.next()
        if i % len_target_train_loader == 0:
            iter_target = iter(target_train_loader)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target = Variable(data_target)

        optimizer.zero_grad()
        label_source_pred, loss_mmd = model(data_source, data_target)
        loss_cls = F.cross_entropy(label_source_pred, label_source.type(torch.long), reduction='mean')
        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        loss = loss_cls + lambd * loss_mmd
        with torch.no_grad():
            train_loss += loss
            pred = label_source_pred.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(label_source.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                epoch, i * len(data_source), len_source_train_dataset,
                100. * i / len_source_train_loader, loss.item(), loss_cls.item(), loss_mmd.item()))
    correct = correct.item()
    correct_rate = correct / len_source_train_dataset
    train_loss = train_loss.item() / len_source_train_loader
    return correct_rate, train_loss


def validate(model):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0

        for data, target in target_valid_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            s_output, _ = model(data)
            test_loss += F.nll_loss(s_output, target.type(torch.long), reduction='mean') # sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss = test_loss.item() / len_target_valid_loader
        correct = correct.item()
        correct_rate = correct / len_target_valid_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            target_name, test_loss, correct, len_target_valid_dataset,
            100. * correct_rate))
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
              source_name, target_name, correct, 100. * correct_rate))
        return correct_rate, test_loss


if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # torch.cuda.set_device(1)
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
    source_train_path = options['source_train_folder']
    target_train_path = options['target_train_folder']
    source_name = 'miccai'
    target_name = "ISBI"
    cuda = not no_cuda and torch.cuda.is_available()

    _, source_train_list, _ = next(os.walk(source_train_path))

    # list scan
    fold = 0
    # fold train data
    df = pd.read_csv(options['target_train_csv_path'])
    # select training scans
    target_train_files = df.loc[df['fold'] != fold, ['patient_id','study']].values
    target_valid_files = df.loc[df['fold'] == fold, ['patient_id', 'study']].values
    target_train_list = [f[0]+f[1] for f in target_train_files]
    target_valid_list = [f[0]+f[1] for f in target_valid_files]

    source_train_list.sort()
    target_train_list.sort()
    target_valid_list.sort()

    source_train_x_data = {f: {m: os.path.join(options['source_train_folder'], f, options['tmp_folder'], n)
                        for m, n in zip(options['modalities'], options['preprocess_x_names'])}
                    for f in source_train_list}
    source_train_y_data = {f: os.path.join(options['source_train_folder'], f, options['tmp_folder'],
                                    options['preprocess_y_names'][0])
                    for f in source_train_list}

    target_train_x_data = {f: {m: os.path.join(options['target_train_folder'], f, options['tmp_folder'], n)
                        for m, n in zip(options['modalities'], options['preprocess_x_names'])}
                    for f in target_train_list}

    target_valid_x_data = {f: {m: os.path.join(options['target_train_folder'], f, options['tmp_folder'], n)
                        for m, n in zip(options['modalities'], options['preprocess_x_names'])}
                    for f in target_valid_list}

    target_valid_y_data = {f: os.path.join(options['target_train_folder'], f, options['tmp_folder'],
                                    options['preprocess_y_names'][0])
                    for f in target_valid_list}

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # to test on small data
    #source_train_x_data={'CHB_train_Case01':source_train_x_data['CHB_train_Case01']}
    #source_train_y_data={'CHB_train_Case01':source_train_y_data['CHB_train_Case01']}
    #target_train_x_data={'training01_01':target_train_x_data['training01_01']}
    #target_valid_x_data={'training01_03':target_valid_x_data['training01_03']}
    #target_valid_y_data={'training01_03':target_valid_y_data['training01_03']}


    source_train_loader = dl.load_training(options, source_train_x_data, source_train_y_data)
    target_train_loader = dl.load_training(options, target_train_x_data, y_data=None)
    target_valid_loader = dl.load_training(options, target_valid_x_data, y_data=target_valid_y_data)

    len_source_train_dataset = len(source_train_loader.dataset)
    len_source_train_loader = len(source_train_loader)

    len_target_train_dataset = len(target_train_loader.dataset)
    len_target_train_loader = len(target_train_loader)

    len_target_valid_dataset = len(target_valid_loader.dataset)
    len_target_valid_loader = len(target_valid_loader)

    model = models.DANNet(num_classes=2)
    print(model)
    if cuda:
        model.cuda()
    model = load_pretrain(model)

    correct = 0
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': lr[1]},
        ], lr=lr[0], momentum=momentum, weight_decay=l2_decay)
    path= os.path.join(options['weight_paths'],options['experiment'])

    Path(path).mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(columns=['lr', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])
    patience = options['patience']
    patience_value = 0
    for epoch in range(1, epochs + 1):
        train_correct, train_loss = train(epoch, model, optimizer)
        #torch.cuda.synchronize()
        t_correct, test_loss = validate(model)

        FILE = os.path.join(path,str(epoch)+'_model.pth')
        torch.save(model, FILE)
        if t_correct > correct:
            correct = t_correct
            patience_value = 0
        else:
            patience_value += 1
        print('patience: ', patience_value)
        #correct = correct.item()
        df = pd.DataFrame([[lr[0], train_loss, train_correct, test_loss,  t_correct]], columns=['lr', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])
        history_df = history_df.append(df)

        history_df.reset_index(inplace=True)
        history_df.drop(columns=['index'], inplace=True)
        history_df.to_csv(options['adapt_history_csv_path'], index=False)
        if patience_value >= patience:
            break
