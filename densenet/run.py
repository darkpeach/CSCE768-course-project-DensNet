import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_load import Protein_Dataset
from dense_net import DenseNet


batch_size = 1
channel_size = 42


use_cuda = torch.cuda.is_available()
def cuda_var_wrapper(var, volatile=False):
    if use_cuda:
        var = Variable(var, volatile=volatile).cuda()
    else:
        var = Variable(var, volatile=volatile)
    return var


class Tool():
    @staticmethod
    def edit_distance(s1, s2):
        size = len(s1)

        match = 0
        for i in range(0, size-1):
            if s1[i] == s2[i]:
                match += 1

        return match / size


cnn = DenseNet(channel_size, batch_size, drop_rate = 0, num_init_features=512)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

if use_cuda:
    cnn.cuda()
    criterion.cuda()


train_dataset = Protein_Dataset(0, 5800)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

# Train the Model
for epoch in range(5):
    for i, (seq, labels) in enumerate(train_loader):
        sequences = cuda_var_wrapper(seq)
        labels = cuda_var_wrapper(labels)

        cat_labels = torch.cat(([labels[i] for i in range(batch_size)]), 1)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(sequences)
        a = outputs.view(9, 700 * batch_size)
        predict_value, predict_index = torch.max(a, 0)
        true_value, true_index = torch.max(cat_labels, 0)

        loss = criterion(a, labels)
        loss.backward()
        optimizer.step()

        distance = Tool.edit_distance(
            predict_index.data.cpu().numpy().tolist(), 
            true_index.data.cpu().numpy().tolist()
        )

        print('Epoch [%d], Step [%d/%d], Loss [%f], Distance [%f]' %(epoch+1, i+1, 5800//batch_size, loss, distance))


#model_path = 'densenet.model2'
#torch.save(cnn.state_dict(), model_path)


validate_dataset = Protein_Dataset(5801, 6133)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=16)


correct = 0.0
total = 0

for i, (seq, labels) in enumerate(validate_loader):
    sequences = cuda_var_wrapper(seq)
    labels = cuda_var_wrapper(labels)

    cat_labels = torch.cat(([labels[i] for i in range(batch_size)]), 1)

    output = cnn(sequences)
    a = output.view(9, 700 * batch_size)
    predict_value, predict_index = torch.max(a, 0)
    true_value, true_index = torch.max(cat_labels, 0)

    distance = Tool.edit_distance(
        predict_index.data.cpu().numpy().tolist(), 
        true_index.data.cpu().numpy().tolist()
    )

    size = len(predict_index.data.cpu().numpy().tolist())
    correct += (distance * size)
    total += size

    sys.stdout.write('\r')
    sys.stdout.write("Step [%d/333]" % (i))
    sys.stdout.flush()

accuracy = correct / total


print("Prediction Accuracy is", accuracy)

