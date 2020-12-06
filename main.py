import os
import torch
from torchvision import datasets
from torchvision.transforms import transforms

import config
from torch.utils.data import DataLoader
from preprocess import SurfaceDataset
from model import Net


if __name__ == "__main__":
    surface_train_dataset = SurfaceDataset(csv_file=os.path.join('Data', 'CombinedOutputs', 'csv', 'all_data.csv'),
                                           root_dir='Data')
    dataloader = DataLoader(surface_train_dataset, batch_size=4,
                            shuffle=True, num_workers=0)
    model = Net()

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for epoch in range(config.epochs):
        running_loss = 0.0

        for i, data in enumerate(dataloader):
            inputs = data['image']
            labels = data['direction']
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
