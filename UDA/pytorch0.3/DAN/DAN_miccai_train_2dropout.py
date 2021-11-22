from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
from config.miccai_settings import *
from utils.data_preprocess import *
import dataset_loader as dl
from torch.utils.tensorboard import SummaryWriter
import sys
from pathlib import Path
from utils.data_load import load_data_patches, generate_data_patches
from DatasetsGeneratorFromFiles import *


def train(epoch, model, optimizer):

    optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)

    model.train()
    train_loss=0
    correct = 0

    #iter_source_train = iter(source_train_loader)
    #num_iter_train = len_source_train_loader
    for i, (data_source_train, label_source_train) in enumerate(train_generator.__getitem__()):
        data_source_train, label_source_train = torch.from_numpy(data_source_train), torch.from_numpy(label_source_train)
    #for i in range(1, num_iter_train):
    #    data_source_train, label_source_train = iter_source_train.next()
        if cuda:
            data_source_train, label_source_train = data_source_train.cuda(), label_source_train.cuda()
        data_source_train, label_source_train = Variable(data_source_train), Variable(label_source_train)

        optimizer.zero_grad()
        label_source_train_pred, _ = model(data_source_train)
        loss = F.cross_entropy(label_source_train_pred, label_source_train.type(torch.long), reduction='mean')

        with torch.no_grad():
            train_loss += loss
            pred = label_source_train_pred.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(label_source_train.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data_source_train), len_source_train_dataset,
                100. * i / len_source_train_loader, loss.item()))

    correct = correct.item()
    correct_rate = correct / len_source_train_dataset
    train_loss = train_loss.item() / len_source_train_loader
    return correct_rate, train_loss


def validate(model):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for i, (data_source_valid, label_source_valid) in enumerate(valid_generator.__getitem__()):
            data_source_valid, label_source_valid = torch.from_numpy(data_source_valid), torch.from_numpy(label_source_valid)
        #iter_source_valid = iter(source_valid_loader)
        #num_iter_valid = len_source_valid_loader
        #for i in range(1, num_iter_valid):
        #    data_source_valid, label_source_valid = iter_source_valid.next()
            if cuda:
                data_source_valid, label_source_valid = data_source_valid.cuda(), label_source_valid.cuda()
            data_source_valid, label_source_valid = Variable(data_source_valid), Variable(label_source_valid)
            s_output, _ = model(data_source_valid)
            test_loss += F.cross_entropy(F.log_softmax(s_output, dim = 1), label_source_valid.type(torch.long)) # sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(label_source_valid.view_as(pred)).cpu().sum()

        test_loss = test_loss.item() / len_source_valid_loader
        correct = correct.item()
        correct_rate = correct / len_source_valid_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            source_name, test_loss, correct, len_source_valid_dataset, 100. * correct_rate))

        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
              source_name, '', correct, 100. * correct_rate))
        return correct_rate, test_loss


if __name__ == '__main__':
    torch.cuda.empty_cache()
    #torch.cuda.synchronize()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # torch.cuda.set_device(1)
    writer = SummaryWriter('runs')

    # Training settings
    st = Settings()
    options = st.get_options()

    second_train = options['second_train']
    pretrained_model = None
    train_count = options['train_count']
    if second_train:
        pretrained_model = models.DANNet(num_classes=2)
        pretrained_model_path = os.path.join(options['weight_paths'], options['experiment'], '1', options['pre_trained_model'])
        pretrained_model = torch.load(pretrained_model_path)

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
    #df = miccai_generate_csv(options)

    # split data to train, validate folds
    #miccai_split_folds(options['train_csv_path'], options['seed'], options['k_fold'])

    # list scan
    fold = 0
    # fold train data
    df = pd.read_csv(options['train_csv_path'])
    # select training scans
    train_files = df.loc[df['fold'] != fold, ['center_id','patient']].values
    valid_files = df.loc[df['fold'] == fold, ['center_id', 'patient']].values
    train_scan_list = [f[0]+'_'+f[1] for f in train_files]
    valid_scan_list = [f[0]+'_'+f[1] for f in valid_files]

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

    if second_train:
        if options['generate_patches']:
            generate_data_patches(train_x_data, train_y_data, options, dataset_name='miccai', model=pretrained_model)
            generate_data_patches(valid_x_data, valid_y_data, options, dataset_name='miccai', model=pretrained_model)
        else:
            pass
    else:
        if options['generate_patches']:
            generate_data_patches(train_x_data, train_y_data, options, dataset_name='miccai')
            generate_data_patches(valid_x_data, valid_y_data, options, dataset_name='miccai')
        else:
            pass
    train_files, train_files_ref, train_patches = load_data_patches(options['h5_path'], options['train_csv_path'], phase='train', fold=fold, options=options)
    train_generator = DatasetGenerator(data=train_files, options=options, patches=train_patches)

    valid_files, valid_files_ref, valid_patches = load_data_patches(options['h5_path'], options['train_csv_path'], phase='valid', fold=fold, options=options)
    valid_generator = DatasetGenerator(data=valid_files, options=options, patches=valid_patches)
    #source_train_loader = dl.load_training(options, train_x_data, train_y_data, model=pretrained_model)
    #source_valid_loader = dl.load_training(options, valid_x_data, valid_y_data, model=pretrained_model)

    # source_test_loader = data_loader.load_testing('', source_path, batch_size, kwargs)

    len_source_train_dataset = train_generator.__len__() * options['batch_size']
    len_source_valid_dataset = valid_generator.__len__() * options['batch_size']
    len_source_train_loader = train_generator.__len__()
    len_source_valid_loader = valid_generator.__len__()

    saved_model=None
    model = models.DANNet(num_classes=2)
    if options['load_initial_weights']:
        model = torch.load(options['initial_weights_file'])
        saved_model = torch.load(options['initial_weights_file'])
        model.load_state_dict(saved_model.state_dict())
        print(options['initial_weights_file'])
    elif options['save_initial_weights']:
        Path(options['initial_weights_path']).mkdir(parents=True, exist_ok=True)
        torch.save(model, options['initial_weights_file'])
    #writer.add_graph(model, torch.rand(size=(128, 2, 16, 16, 16)))
    #writer.flush()
    #writer.close()
    # sys.exit()
    correct = 0
    print(model)
    if cuda:
        model.cuda()

    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': lr[1]},
        ], lr=lr[0], momentum=momentum, weight_decay=l2_decay)
    path= os.path.join(options['weight_paths'], options['experiment'], train_count)

    Path(path).mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(columns=['lr', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])
    patience = options['patience']
    patience_value = 0
    for epoch in range(1, epochs + 1):
        train_correct, train_loss = train(epoch, model, optimizer)
        # torch.cuda.synchronize()
        t_correct, test_loss = validate(model)

        FILE = os.path.join(path,str(epoch)+'_model.pth')
        torch.save(model, FILE)
        if t_correct > correct:
            correct = t_correct
            patience_value = 0
        else:
            patience_value += 1
        print('patience: ', patience_value)
        # correct = correct.item()
        df = pd.DataFrame([[optimizer.param_groups[0]['lr'], train_loss, train_correct, test_loss,  t_correct]], columns=['lr', 'loss', 'accuracy', 'val_loss', 'val_accuracy'])
        history_df = history_df.append(df)

        history_df.reset_index(inplace=True)
        history_df.drop(columns=['index'], inplace=True)
        history_df.to_csv(options['history_csv_path'], index=False)
        if patience_value >= patience:
            break
