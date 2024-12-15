import torch
import numpy as np
from measurement import generate_mask, gaussian_noise, poission_noise, inpainting, downsample, colorization
from cifar10_loader import dataloader
import os 
import matplotlib.pylab as plt
from tqdm.auto import tqdm
from diffusers import DDPMPipeline

device = torch.device("cuda")  
model = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)


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
        self.fixed_log_post_var = torch.log(self.betas * (1.0 - self.alpha_i_1) / (1.0 - self.alpha_i))
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
            s = model.unet(x_i.float(), t).sample
            mu = s
            x_0 = self.get_x_0(x_i, mu, t)
            x_0 = x_0.clamp(-1, 1)
            var = self.fixed_log_post_var.to(device)[t]
            x_i_1 = self.get_x_i_1(x_i, x_0, var, t)
            x_i_1 = self.apply_dps(y, x_0, x_i, x_i_1)
            x_i = x_i_1.detach_()
            # remove if you don't want to print every 10 iterations.
            # if i % 10 == 0:
            #     file_path = os.path.join('./results/', f"progress/{str(i).zfill(4)}.png")
            #     plt.imsave(file_path, clear_img(x_i))
        
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
    method = "super_resolution"
    if method == "inpainting":
        operator = inpainting
    elif method == "super_resolution":
        operator = downsample
    elif method == 'colorization':
        operator = colorization
    else:
        operator = None

    os.makedirs(f'./results/{method}/start/', exist_ok=True)
    os.makedirs(f'./results/{method}/final/', exist_ok=True)
    
        # Experiment configurations for 5 iterations
    experiments = [
        {"sigma": 0.05, "scale": 0.1}, # Experiment 1: sigma=0.01
        {"sigma": 0.05, "scale": 1.0}, # Experiment 1: sigma=0.01
        {"sigma": 0.05, "scale": 0.3},  # Experiment 2: default
        {"sigma": 0.05, "scale": 0.5}  # Experiment 3: default
        # {"sigma": 0.05, "scale": 0.1},  # Experiment 4: default
        # {"sigma": 0.05, "scale": 0.3},  # Experiment 5: scale=0.1
    ]

    for exp_idx, config in enumerate(experiments, start=1):
        sigma = config["sigma"]
        scale = config["scale"]

        # Format sigma and scale for file naming
        sigma_str = f"sigma{int(sigma * 1000):03d}"  # Example: 0.01 -> sigma001
        scale_str = f"scale{int(scale * 100):02d}"   # Example: 0.1 -> scale01


        dps = DPS(num_timesteps=1000, 
                scale=scale, # Something to note here is that in the paper, if you look at the appendix you will find experiments details which proivde different scale values to the one used in their configs file. I used the one in the configs file.
                noise="gaussian",
                sigma=sigma, 
                lamb=1,
                method=method,
                operator=operator)

        x = torch.randn([1,3,32,32]).to(device)
        for i, (X , _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if exp_idx < 4:
                continue
            x = torch.randn([1,3,32,32]).to(device).requires_grad_()
            if method == "inpainting": 
                dps.mask = generate_mask(X, 256, 128).to(device)
                y = dps.operator(X.to(device), dps.mask)
            elif method == "super_resolution":
                # Downsample HR image to create LR input
                X = X.to(device)
                y = downsample(X, scale_factor=0.25)
            elif method == 'colorization':
                X = X.to(device)
                y = colorization(X)
            else:
                y = dps.operator(X.to(device), dps.mask)

            
            if dps.noise == "gaussian":
                y = gaussian_noise(y, dps.sigma).requires_grad_()
            else:
                y = poission_noise(y, dps.lamb).requires_grad_()
            # Save the starting result
            file_path = os.path.join(f'./results/{method}/start/exp{exp_idx}_{sigma_str}_{scale_str}_{str(i).zfill(5)}.png')
            plt.imsave(file_path, clear_img(y))
            
            # Reconstruct and save the final result
            result = dps.reverse(x, y)  # Data sent to DPS to reverse it
            file_path = os.path.join(f'./results/{method}/final/exp{exp_idx}_{sigma_str}_{scale_str}_{str(i).zfill(5)}.png')
            plt.imsave(file_path, clear_img(result))    

if __name__ == '__main__':
    main()