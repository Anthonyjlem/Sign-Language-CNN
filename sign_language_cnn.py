from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

def load_data(data):
    df = data.copy()
    labels = df["label"]
    labels = labels.values
    df.drop("label", axis=1, inplace=True)
    data = df.values
    return torch.tensor(data).float(), torch.tensor(labels).long()

def convert_to_2d(data, dim):
    data = data.reshape(data.shape[0], dim, dim)
    data = data.unsqueeze(1)
    data = data.repeat(1, 3, 3, 3)
    return data

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print("Training on GPU")
else:
    print("Training on CPU")

# load in data

valid_size = 0.2 # percentage of training data for validation

raw_train_data = pd.read_csv("input/sign_mnist_train.csv", sep=",")
raw_test_data = pd.read_csv("input/sign_mnist_test.csv", sep=",")
test_data = raw_test_data.truncate(after=7099)
train_data, valid_data = train_test_split(raw_train_data, test_size=valid_size)

train_data, train_labels = load_data(train_data)
valid_data, valid_labels = load_data(valid_data)
test_data, test_labels = load_data(test_data)
print(test_data.shape[0])

# visualize an image

image = train_data[0].reshape(28, 28)
plt.imshow(image)
print(train_labels[0])

# convert data to proper shape
    
train_data = convert_to_2d(train_data, 28)
valid_data = convert_to_2d(valid_data, 28)
test_data = convert_to_2d(test_data, 28)
print(train_data[0].shape)

# define CNN

model = models.alexnet(pretrained=True)
print(model)

# update model for our uses

# freezing parameters so they aren't updated
for param in model.parameters():
    param.requires_grad = False

# building new classifier

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(9216, 2048)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(2048, 512)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc3', nn.Linear(512, 128)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc4', nn.Linear(128, 26)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

if train_on_gpu:
    model.cuda()

# loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

# train the model

epochs = 50
batch_size = 100
valid_loss_min = np.Inf # retain the lowest validation loss

for e in range(epochs):
    
    train_loss = 0
    valid_loss = 0
    
    # training
    
    model.train()
    
    for i in range(0, train_data.shape[0], batch_size):
        data = train_data[i:i+batch_size]
        target = train_labels[i:i+batch_size]
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        
    # validation
    
    model.eval()
    for i in range(0, valid_data.shape[0], batch_size):
        data = valid_data[i:i+batch_size]
        target = valid_labels[i:i+batch_size]
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
    
    # average losses
    train_loss  = train_loss/train_data.shape[0]
    valid_loss = valid_loss/valid_data.shape[0]
    
    print("Epoch", e+1, "\tTraining Loss:", train_loss, "\tValidation Loss:", valid_loss)
    
    # save model if performing better
    if valid_loss < valid_loss_min:
        print("Saving model...")
        torch.save(model.state_dict(), "sign_language_model.pt")
        valid_loss_min = valid_loss

# load best model
model.load_state_dict(torch.load("sign_language_model.pt"))

# test the CNN

batch_size = 100
test_loss = 0
class_correct = list(0 for i in range(26))
class_total = list(0 for i in range(26))

model.eval()

for i in range(0, test_data.shape[0], batch_size):
    data = test_data[i:i+batch_size]
    target = test_labels[i:i+batch_size]
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    
    # get prediction
    _, pred = torch.max(output, 1)
    
    # compare prediction to target
    correct_tensor = pred.eq(target.data.view_as(pred))
    if train_on_gpu:
        correct = np.squeeze(correct_tensor.cpu().numpy())
    else:
        correct = np.squeeze(correct_tensor.numpy())
    
    # keep track of results
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss/test_data.shape[0]
print("Test Loss:", test_loss)  

# print accuracy for each class

alphabet = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n',
            14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z'}

for i in range(26):
    if class_total[i] > 0:
        print("Test accuracy of " + alphabet[i] + "\'s: {:.1f}%".format(class_correct[i]/class_total[i]*100))
    else:
        print("Test accuracy of " + alphabet[i] + "\'s: N/A")
    
print("Overall accuracy: {:.2f}%".format(np.sum(class_correct)/np.sum(class_total)*100))
