# Reproducing-Diffusion-Posterior-Sampling

In this project, we are reproducing and investigating the results from the paper "Diffusion Posterior Sampling for General Noisy Inverse Problems". The method is tested on both the CIFAR-10 and the FFHQ datasets on the following: inpainting, super-resolution, Gaussian blur, non-linear blur and colorization. Other experiments included studying the effect of the step size in the gradient step and the level of noise applied to the images. 

To run the non-linear blur experiment:
```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
```

Link to paper: https://arxiv.org/abs/2209.14687

