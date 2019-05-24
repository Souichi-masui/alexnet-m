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

    valset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                            download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True)
    classes = ("plane", "car", "bird", "cat",
            "deer", "dog", "frog", "horse", "ship", "truck")

    # model = alexnet.AlexNet()
    # net = "alexnet"
    model = alextnetfixed.AlexNetfixed()
    net = "alexnet-fixed"
    model.load_state_dict(torch.load("save/weight(" + net + ").pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    mode = "val"
    epoch_num = 60
    logs = pd.DataFrame(columns=["epochs", "loss", "accuracy"])
    

    for epoch in range(epoch_num):  
        model.eval()
        tloss = 0.0
        tcorrect = 0.0
        total = 0.0
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            tcorrect += (predicted == labels).sum().item()
            tloss += loss.item()
        print("Epoch_num:%3d, Loss: %2.4f, Accuracy: %2.4f %%" % (epoch + 1, tloss / len(valloader), (tcorrect / total) * 100))
        tmp_se = pd.Series([epoch + 1, tloss / len(valloader), (tcorrect / total) * 100], index=logs.columns )
        logs = logs.append(tmp_se, ignore_index=True )
    end_time = time.time() - start_time
    logs.to_csv("save/" + net + "_" + mode + "_loss_accuracy.csv", index=False)

    torch.save(model.state_dict(),"save/weight(" + net + ").pth")
    print("Time:%ds \nTraining Finished" %(end_time))