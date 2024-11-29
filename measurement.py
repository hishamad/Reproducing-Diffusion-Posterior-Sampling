import torch

def generate_mask(X, img_dim, box_size, margin_size=5):
    x = torch.randint(low=margin_size, high=img_dim-box_size-margin_size, size=[1])
    y = torch.randint(low=margin_size, high=img_dim-box_size-margin_size, size=[1])
    mask = torch.ones(X.size(), device=X.device)
    mask[..., x:x+box_size, y:y+box_size] = 0
    return mask

def gaussian_noise(X, sigma=0.05):
    return torch.randn_like(X, device=X.device) * sigma  + X