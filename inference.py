import torch
from utils.unet import create_model
import numpy as np
from measurement import generate_mask, gaussian_noise, poission_noise, inpainting
from loader import dataloader
import os 
import matplotlib.pylab as plt
from tqdm.auto import tqdm

device = torch.device("cuda")  
print(device)
model = create_model(   
                        image_size=256,
                        num_channels=128,
                        num_res_blocks=1,
                        channel_mult="",
                        learn_sigma=True,
                        class_cond=False,
                        use_checkpoint=False,
                        attention_resolutions=16,
                        num_heads=4,
                        num_head_channels=64,
                        num_heads_upsample=-1,
                        use_scale_shift_norm=True,
                        dropout=0,
                        resblock_updown=True,
                        use_fp16=False,
                        use_new_attention_order=False,
                        model_path='models/ffhq_10m.pt',
                    ).to(device).eval()

print("Model loaded!")

class DPS:
    def __init__(self, num_timesteps, scale=0.5, noise="gaussian", sigma=0.05, lamb=1, method="inpainting", operator=None):
        self.num_timesteps = num_timesteps
        self.scale = scale
        self.noise = noise
        self.sigma = sigma
        self.lamb = lamb
        scale = 1000 / self.num_timesteps
        self.betas = torch.linspace(scale * 0.0001, scale * 0.02, num_timesteps, dtype=torch.float64)
        self.alphas = 1 - self.betas
        self.alpha_i = torch.cumprod(self.alphas, axis=0)
        self.alpha_i_1 = torch.cat((torch.tensor([1.0], dtype=torch.float64), self.alpha_i[:-1]))
        self.alpha_recip = torch.sqrt(1.0 / self.alpha_i)
        self.alpha_recip_1 = torch.sqrt(1.0 / self.alpha_i - 1)
        self.x_i_coeff = (torch.sqrt(self.alphas)  * (1-self.alpha_i_1)) / (1-self.alpha_i)
        self.x_0_coeff = (torch.sqrt(self.alpha_i_1) * self.betas) / (1-self.alpha_i)
        posterior_var = self.betas * (1.0 - self.alpha_i_1) / (1.0 - self.alpha_i)
        self.log_post_var = torch.log(torch.cat((posterior_var[1:2], posterior_var[1:])))
        self.operator = operator
        self.method = method
        self.mask = None

    def get_range_var(self, var, t):
        f = (var + 1.0) / 2.0
        mini = self.log_post_var.to(device)[t]
        maxi = torch.log(self.betas).to(device)[t]
        range_var = f * maxi + (1-f) * mini
        return range_var

    def get_x_0(self, x, s, t):
        # From paper:
        # alpha_recip = self.alpha_recip.to(device)[t]
        # alpha_i = self.alpha_i.to(device)[t]
        
        # return alpha_recip * (x + (1-alpha_i)*s)
        alpha_recip = self.alpha_recip.to(device)[t]
        alpha_recip_1 = self.alpha_recip_1.to(device)[t]
        return alpha_recip * x - alpha_recip_1 * s 
    
    def get_x_i_1(self, x_i, x_0, var, t):
        x_i_coeff = self.x_i_coeff.to(device)[t]
        x_0_coeff = self.x_0_coeff.to(device)[t]
        z = torch.randn_like(x_i)
        if t != 0:
            sigma = torch.exp(0.5 * var)
        else:
            sigma = 0
        
        return x_i * x_i_coeff + x_0_coeff * x_0 + sigma * z

    def apply_dps(self, y, x_0, x_i, x_i_1):
        if self.noise == "gaussian":
            if self.method != "inpainting": # If it is not inpainting
                diff_norm = torch.linalg.norm(y - self.operator(x_0))
            else:
                diff_norm = torch.linalg.norm(y - (self.mask * x_0))
            norm_grad = torch.autograd.grad(outputs=diff_norm, inputs=x_i)[0]

        else: # if poisson
            if self.method != "inpainting": # If it is not inpainting
                diff_norm = torch.linalg.norm(y - self.operator(x_0)) / y.abs()
            else:
                diff_norm = torch.linalg.norm(y - (self.mask * x_0)) / y.abs()
            norm_grad = torch.autograd.grad(outputs=diff_norm.mean(), inputs=x_i)[0]   
        x_i_1 -= self.scale * norm_grad 
        return x_i_1
    
    def reverse(self, x, y):
        x_i = x
        for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps):
            t = torch.tensor([i]).to(device)
            x_i = x_i.requires_grad_()
            s = model(x_i.float(), t)
            mu, var = torch.split(s, split_size_or_sections=x_i.shape[1], dim=1)
            x_0 = self.get_x_0(x_i, mu, t)
            x_0 = x_0.clamp(-1, 1)
            range_var = self.get_range_var(var, t)
            x_i_1 = self.get_x_i_1(x_i, x_0, range_var, t)
            x_i_1 = self.apply_dps(y, x_0, x_i, x_i_1)
            x_i = x_i_1.detach_()
            # remove if you don't want to print every 10 iterations.
            if i % 10 == 0:
                file_path = os.path.join('./results/', f"progress/{str(i).zfill(4)}.png")
                plt.imsave(file_path, clear_img(x_i))
        
        return x_i

# From the paper code:
def clear_img(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    img = np.transpose(x, (1, 2, 0))
    img -= np.min(img)
    img /= np.max(img)
    return img



def main():
    # Specifiy measurement method
    method = "inpainting"
    if method == "inpainting":
        operator = inpainting
    elif method == "":
        operator = None
    else:
        operator = None

    # Specifiy DPS config here
    dps = DPS(num_timesteps=1000, 
            scale=0.5, 
            noise="gaussian",
            sigma=0.05,
            lamb=1,
            method=method,
            operator=operator)

    x = torch.randn([1,3,256,256]).to(device)
    for i, X in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = torch.randn([1,3,256,256]).to(device).requires_grad_()
        if method == "inpainting":
            dps.mask = generate_mask(X, 256, 128).to(device)
            y = dps.operator(X.to(device), dps.mask)
        else:
            y = dps.operator(X.to(device), dps.mask)
        
        if dps.noise == "gaussian":
            y = gaussian_noise(y, dps.sigma).requires_grad_()
        else:
            y = poission_noise(y, dps.lamb).requires_grad_()
        file_path = os.path.join('./results/', f"start/{str(i).zfill(5)}.png")
        plt.imsave(file_path, clear_img(y))
        result = dps.reverse(x, y)
        file_path = os.path.join('./results/', f"final/{str(i).zfill(5)}.png")
        plt.imsave(file_path, clear_img(result))       

if __name__ == '__main__':
    main()