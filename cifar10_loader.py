import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

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