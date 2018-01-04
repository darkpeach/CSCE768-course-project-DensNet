import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_load import ProteinDatasets

batch_size = 58
channel_size = 42
test_dataset = ProteinDatasets(0, 5800)
train_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)


class Tool:
    @staticmethod
    def edit_distance(s1, s2):
        size = len(s1)

        match = 0
        for i in range(0, size-1):
            if s1[i] == s2[i]:
                match += 1

        return match / size


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(channel_size, 16, kernel_size=3, padding=1),
            #nn.BatchNorm1d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            #nn.BatchNorm1d(32),
            nn.ReLU())
        self.fc = nn.Linear(32*700, 500)
        self.fc2 = nn.Linear(500, 700*9)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    cnn = CNN()

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    # Train the Model
    for epoch in range(100):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)

            cat_labels = torch.cat(([labels[i] for i in range(batch_size)]), 1)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            a = outputs.view(9, 700 * batch_size)
            predict_value, predict_index = torch.max(a, 0)
            true_value, true_index = torch.max(cat_labels, 0)

            loss = criterion(a, labels)
            loss.backward()
            optimizer.step()

            # time1 = time.time()
            # print(predict_index.data.numpy().shape)
            # print(true_index.data.numpy().shape)
            distance = Tool.edit_distance(predict_index.data.numpy().tolist(), true_index.data.numpy().tolist())
            # time2 = time.time()
            # print(time2-time1)

            print('Epoch [%d], Step [%d/%d], Loss [%f], Distance [%f]' %(epoch+1, i+1, 5800//batch_size, loss, distance))
