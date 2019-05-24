import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import TestModel
import cv2
import AlexNet

if __name__ == '__main__':

    model = TestModel.TestModel()
    model.load_state_dict(torch.load("weight(Normal).pth"))
    model.eval()
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    x = torch.zeros(4,3,32,32)
    img = cv2.imread("apple.jpg") 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(32,32))

    img = (img - float(128))/ float(255)
    img = img.transpose(2,0,1)
    img = img.astype(np.float32)
    data = torch.from_numpy(img[np.newaxis, :, :, :])
    x = x + data[0]
    outputs = model(data)
    _, predicted = torch.max(outputs, 1)

    print("Predicted: ", " ".join("%5s" % classes[predicted[0]]))