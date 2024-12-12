import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import os


torch.manual_seed(10)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 1

dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_images = 100
random_indices = torch.randperm(len(dataset))[:num_images]
subset = Subset(dataset, random_indices)

dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

torch.seed()

os.makedirs('./cifar10')

for i, (X, _) in enumerate(dataloader):  
    img = X[0] / 2 + 0.5 
    npimg = img.numpy() 
    plt.imsave(f"./cifar10/{str(i).zfill(5)}.png", np.transpose(npimg, (1, 2, 0)))  