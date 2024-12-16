import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.ndimage
import yaml
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
import scipy.ndimage

def generate_mask(X, img_dim, box_size, margin_size=5):
    x = torch.randint(low=margin_size, high=img_dim-box_size-margin_size, size=[1])
    y = torch.randint(low=margin_size, high=img_dim-box_size-margin_size, size=[1])
    mask = torch.ones(X.size(), device=X.device)
    mask[..., x:x+box_size, y:y+box_size] = 0
    return mask

def inpainting(X, mask):
    return X * mask

def gaussian_noise(X, sigma=0.05):
    return torch.randn_like(X, device=X.device) * sigma  + X

def poission_noise(X, lamb=1):
    X = (X + 1.0) / 2.0
    X = X.clamp(0, 1)
    X = (torch.poisson(X * 255.0 * lamb) / (255.0 * lamb)) * 2.0 - 1.0
    return X.clamp(-1, 1)

def downsample(x, scale_factor=0.25):
    return F.interpolate(x, scale_factor=scale_factor, mode='bicubic', align_corners=False)


def colorization(x):
    R = x[:, 0:1, :, :]
    G = x[:, 1:2, :, :]
    B = x[:, 2:3, :, :]

    gray = 0.299 * R + 0.587 * G + 0.114 * B
    
    gray_3ch = gray.repeat(1, 3, 1, 1)

    return gray_3ch  # shape: [N, 1, H, W]


def create_gaussian_kernel(kernel_size, sigma):
    center = kernel_size // 2
    grid = np.zeros((kernel_size, kernel_size))
    grid[center, center] = 1.0
    gaussian_kernel = scipy.ndimage.gaussian_filter(grid, sigma=sigma)
    return torch.from_numpy(gaussian_kernel).float()

# From original paper code
class GaussianBlurLayer(nn.Module):
    def __init__(self, kernel_size=31, sigma=3.0):
        """
        Gaussian Blur Layer that applies a pre-defined Gaussian kernel.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Reflection padding to preserve spatial dimensions
        padding_size = kernel_size // 2
        self.padding = nn.ReflectionPad2d(padding_size)

        # Convolution with group=3 to apply the same kernel to each RGB channel
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=3,
            bias=False,
        )

        # Initialize Gaussian weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Create Gaussian kernel
        gaussian_kernel = create_gaussian_kernel(self.kernel_size, self.sigma)
        kernel_tensor = gaussian_kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.conv.weight = nn.Parameter(kernel_tensor, requires_grad=False)

    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        return x

# From original paper code
class GaussianBlurOperator:
    def __init__(self, kernel_size=31, sigma=3.0, device=None):
        """
        Operator for applying Gaussian blur using the GaussianBlurLayer.
        """
        self.device = device if device else torch.device("cpu")
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Initialize the Gaussian blur layer
        self.blur_layer = GaussianBlurLayer(kernel_size, sigma).to(self.device)

    def forward(self, input_data):
        """
        Apply Gaussian blur to input data.
        """
        return self.blur_layer(input_data)

# From original paper code
class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 


class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred