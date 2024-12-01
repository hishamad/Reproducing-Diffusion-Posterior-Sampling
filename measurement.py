import torch

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