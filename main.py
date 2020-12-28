import copy
import os
import time
import argparse

import config
import datetime
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from preprocess import SurfaceDataset
from model import Net


def train_val_split(dataset):
    train_len = int(len(dataset) * (1 - config.val_split))
    val_len = int(len(dataset) * config.val_split)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    return train_set, val_set


def train_model(model, dataloaders, criterion, optimizer, tb_writer):
    since = time.time()

    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(1, config.epochs + 1):
        print('Epoch {}/{}'.format(epoch, config.epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs = data['image'].to(args.device)
                labels = data['surface'].to(args.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                print('Best val Loss: {:4f}'.format(best_loss))

            tb_writer.add_scalar('Loss/' + phase, epoch_loss, epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()
    args.device = None

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    tb_writer = SummaryWriter('logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    dataset = SurfaceDataset(csv_file=os.path.join('Data', 'CombinedOutputs', 'csv', 'all_data.csv'),
                             root_dir='Data')

    train_set, val_set = train_val_split(dataset)

    dataloaders = {}
    dataloaders['train'] = DataLoader(train_set, batch_size=4,
                            shuffle=True, num_workers=4)
    dataloaders['val'] = DataLoader(val_set, batch_size=4,
                            shuffle=True, num_workers=4)

    model = Net().to(args.device)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    train_model(model, dataloaders, criterion, optimizer, tb_writer)