import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import alexnet
import alextnetfixed
import time
import pandas as pd

if __name__ == '__main__':
    start_time = time.time()
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(size=32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    classes = ("plane", "car", "bird", "cat",
            "deer", "dog", "frog", "horse", "ship", "truck")

    # model = alexnet.AlexNet()
    model = alextnetfixed.AlexNetfixed()
    net = "alexnet-fixed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    mode = "Train"
    epoch_num = 60
    logs = pd.DataFrame(columns=["epochs", "loss", "accuracy"])
    for epoch in range(epoch_num):  
        model.train()
        tloss = 0.0
        tcorrect = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            tcorrect += (predicted == labels).sum().item()
            tloss += loss.item()
        print("Epoch_num:%3d, Loss: %2.4f, Accuracy: %2.4f %%" % (epoch + 1, tloss / len(trainloader), (tcorrect / total) * 100))
        tmp_se = pd.Series([epoch + 1, tloss / len(trainloader), (tcorrect / total) * 100], index=logs.columns )
        logs = logs.append(tmp_se, ignore_index=True )
    end_time = time.time() - start_time
    logs.to_csv("save/" + net + "_" + mode + "_loss_accuracy.csv", index=False)

    torch.save(model.state_dict(),"save/weight(" + net + ").pth")
    print("Time:%ds \nTraining Finished" %(end_time))