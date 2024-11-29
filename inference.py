import torch
from utils.unet import create_model
import numpy as np
from measurement import generate_mask, gaussian_noise
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


def get_betas_alpha(num_timesteps):
    scale = 1000 / num_timesteps
    betas = torch.linspace(scale * 0.0001, scale * 0.02, num_timesteps, dtype=torch.float64)
    alphas = 1 - betas
    alpha_i = torch.cumprod(alphas, axis=0)
    alpha_i_1 = torch.cat((torch.tensor([1.0], dtype=torch.float64), alpha_i[:-1]))
    alpha_recip = torch.sqrt(1.0 / alpha_i)
    alpha_recip_1 = torch.sqrt(1.0 / alpha_i - 1)
    return betas, alphas, alpha_i, alpha_i_1, alpha_recip, alpha_recip_1

def get_coeff(betas, alphas, alpha_i, alpha_i_1):
    x_i_coeff = (torch.sqrt(alphas)  * (1-alpha_i_1)) / (1-alpha_i)
    x_0_coeff = (torch.sqrt(alpha_i_1) * betas) / (1-alpha_i)
    return x_i_coeff, x_0_coeff

def get_x_0(x, s, t, alpha_i, alpha_recip, alpha_recip_1):
    # From paper:
    # alpha_recip = alpha_recip.to(device)[t]
    # alpha_i = alpha_i.to(device)[t]
    
    # return alpha_recip * (x + (1-alpha_i)*s)
    alpha_recip = alpha_recip.to(device)[t]
    alpha_recip_1 = alpha_recip_1.to(device)[t]
    return alpha_recip * x - alpha_recip_1 * s 
    
def get_x_i_1(x_i, x_0, x_i_coeff, x_0_coeff, var, t):
    x_i_coeff = x_i_coeff.to(device)[t]
    x_0_coeff = x_0_coeff.to(device)[t]
    z = torch.randn_like(x_i)
    if t != 0:
        sigma = torch.exp(0.5 * var)
    else:
        sigma = 0
    
    return x_i * x_i_coeff + x_0_coeff * x_0 + sigma * z

def gaussian_dps(y, mask, x_0, x_i, x_i_1):
    diff_norm = torch.linalg.norm(y - (mask * x_0))
    norm_grad = torch.autograd.grad(outputs=diff_norm, inputs=x_i)[0]
    scale = 0.5
    x_i_1 -= scale * norm_grad 
    return x_i_1

# From the paper code:
def clear_img(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    img = np.transpose(x, (1, 2, 0))
    img -= np.min(img)
    img /= np.max(img)
    return img

def get_log_post_var(betas, alpha_i, alpha_i_1):
    posterior_var = betas * (1.0 - alpha_i_1) / (1.0 - alpha_i)
    log_post_var = torch.log(torch.cat((posterior_var[1:2], posterior_var[1:])))
    return log_post_var

def get_range_var(betas, log_post_var, var, t):
    f = (var + 1.0) / 2.0
    mini = log_post_var.to(device)[t]
    maxi = torch.log(betas).to(device)[t]
    range_var = f * maxi + (1-f) * mini
    return range_var

def reverse(x, y, num_timesteps, mask):
    x_i = x
    
    for i in tqdm(reversed(range(num_timesteps))):
        t = torch.tensor([i]).to(device)
        x_i = x_i.requires_grad_()
        s = model(x_i.float(), t)
        
        mu, var = torch.split(s, split_size_or_sections=x_i.shape[1], dim=1)
        x_0 = get_x_0(x_i, mu, t, alpha_i, alpha_recip, alpha_recip_1)
        x_0 = x_0.clamp(-1, 1)
        range_var = get_range_var(betas, log_post_var, var, t)
        x_i_1 = get_x_i_1(x_i, x_0, x_i_coeff, x_0_coeff, range_var, t)
        x_i_1 = gaussian_dps(y, mask, x_0, x_i, x_i_1)
        x_i = x_i_1.detach_()
        # 
        if i % 10 == 0:
            file_path = os.path.join('./results/', f"progress/{str(i).zfill(4)}.png")
            plt.imsave(file_path, clear_img(x_i))
    
    return x_i

num_timesteps = 1000
betas, alphas, alpha_i, alpha_i_1, alpha_recip, alpha_recip_1 = get_betas_alpha(num_timesteps)
x_i_coeff, x_0_coeff = get_coeff(betas, alphas, alpha_i, alpha_i_1)
log_post_var = get_log_post_var(betas, alpha_i, alpha_i_1)
x = torch.randn([1,3,256,256]).to(device)

def main():
    for i, X in enumerate(dataloader):
        x = torch.randn([1,3,256,256]).to(device).requires_grad_()
        mask = generate_mask(X, 256, 128).to(device)
        y = mask * X.to(device)
        y = gaussian_noise(y).requires_grad_()
        file_path = os.path.join('./results/', f"start/{str(i).zfill(5)}.png")
        plt.imsave(file_path, clear_img(y))
        result = reverse(x, y, num_timesteps, mask)
        file_path = os.path.join('./results/', f"final/{str(i).zfill(5)}.png")
        plt.imsave(file_path, clear_img(result))       

if __name__ == '__main__':
    main()