import time
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import models
from itertools import chain
from PIL import Image
import resource

WARM_UP_ROUND=0
TEST_ROUND=1
MODEL={}
MODEL[0]="cifar10_mobilenetv2_x0_5"
MODEL[1]="cifar10_shufflenetv2_x0_5"
BATCH_SIZE=1

def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)
            z = model.forward(inputs)

            batch_loss = loss_fn(z, labels)
            val_loss_cum += batch_loss.item()
            hard_preds = torch.argmax(z,dim=1)
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            val_acc_cum += acc_batch_avg
    return val_loss_cum/len(val_loader), val_acc_cum/len(val_loader)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#test dataset choose cifar
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

#test 2 model one time
for j in range(2):
    #pretrained model download
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", MODEL[j], pretrained=True)

    loss_fn=nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    #we need to warm up before the real test
    for i in range(WARM_UP_ROUND):
        validate(model, loss_fn, testloader, device)

    #real test
    average_time=0.0
    for i in range(TEST_ROUND):
        prev_timestamp = time.time()
        val_loss, val_acc = validate(model, loss_fn, testloader, device)
        timenow=time.time()
        average_time+=(timenow-prev_timestamp)

    average_time/=TEST_ROUND
    fps=10000/average_time
    MemU=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print('Model Name'+MODEL[j])
    print(f"Val. loss: {val_loss:.3f}, ")
    print(f"Val. acc.: {val_acc:.3f}, ")
    print(f"Time spend: {average_time:.3f}")
    print(f"FPS.: {fps:.3f}, ")
    print(f"MemU.: {MemU:.3f}, ")

