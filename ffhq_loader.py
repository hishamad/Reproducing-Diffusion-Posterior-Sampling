from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from glob import glob
from PIL import Image

class FFHQ(VisionDataset):
    def __init__(self, root, transforms):
        super().__init__(root, transforms)
        self.paths = sorted(glob(root + '/**/*.png', recursive=True))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        img = self.transforms(img)
        return img
    
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = FFHQ('./data/', transforms) 
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

