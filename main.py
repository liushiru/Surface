import copy
import os
import time

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


def train_model(model, dataloaders, criterion, optimizer, tb_writer, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(config.epochs):
        print('Epoch {}/{}'.format(epoch, config.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(torch.device)
                labels = labels.to(torch.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            tb_writer.add_scalar('Acc/' + phase, epoch_acc, epoch)
            tb_writer.add_scalar('Loss/' + phase, epoch_loss, epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == "__main__":

    tb_writer = SummaryWriter('logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    dataset = SurfaceDataset(csv_file=os.path.join('Data', 'CombinedOutputs', 'csv', 'all_data.csv'),
                             root_dir='Data')

    train_set, val_set = train_val_split(dataset)

    dataloaders = {}
    dataloaders['train'] = DataLoader(train_set, batch_size=4,
                            shuffle=True, num_workers=0)
    dataloaders['val'] = DataLoader(val_set, batch_size=4,
                            shuffle=True, num_workers=0)

    # dataloader = DataLoader(dataset, batch_size=4,
    #                         shuffle=True, num_workers=0)
    model = Net()

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    train_model(model, dataloaders, criterion, optimizer, tb_writer, is_inception=False)

    # for epoch in range(config.epochs):
    #     accumulate_loss = 0.0
    #     running_loss = 0.0
    #
    #     for i, data in enumerate(dataloader):
    #         inputs = data['image']
    #         labels = data['direction']
    #         optimizer.zero_grad()
    #
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #         accumulate_loss += loss.item()
    #
    #         if i % 10 == 9:
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss / 10))
    #             running_loss = 0.0
    #
    #     writer.add_scalar('Loss/train', accumulate_loss, epoch)
    #     print(running_loss)