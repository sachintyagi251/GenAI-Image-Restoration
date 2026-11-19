# Project Report Outline

## Title
Deep Generative Image Restoration Using Variational Autoencoders and Diffusion Models

## Abstract
This project presents a web-based image restoration application designed to reduce image degradation such as noise and blur. The system explores classical image filters, a Denoising Autoencoder, and a Variational Autoencoder. The application accepts user images, processes them using selected methods, displays before-and-after comparisons, and evaluates results with MSE, PSNR and SSIM when aligned reference images are available. A diffusion-based image-to-image module is proposed as an optional generative enhancement. The prototype is implemented in Python, PyTorch, OpenCV and Streamlit.

## Problem Statement
Images captured under poor lighting, low-quality sensors, transmission errors or aging can contain noise and degradation. The project aims to develop an accessible application that demonstrates deep-learning-based restoration and compares its outputs with traditional filtering techniques.

## Objectives
1. Build a user-friendly image-restoration web interface.
2. Train a Denoising Autoencoder on synthetic noisy-clean image pairs.
3. Implement a VAE for image reconstruction and latent representation learning.
4. Compare classical filters with learned models.
5. Evaluate restoration using MSE, PSNR and SSIM on aligned reference pairs.
6. Explore generative restoration using an optional diffusion model.
7. Discuss model limitations, hallucination, bias and responsible use.

## Methodology
Dataset -> preprocessing and resize -> create synthetic Gaussian-noise pairs -> train DAE/VAE -> inference -> visual comparison -> reference-based evaluation -> analysis.

## Model formulas
MSE: L = (1/N) sum (x_i - y_i)^2

VAE objective: L_VAE = L_reconstruction + beta * D_KL(q(z|x) || p(z))

PSNR: 10 log10(MAX^2 / MSE)

## Expected Results
- Noise reduction in synthetic test images.
- Reconstructed outputs from trained models.
- Qualitative comparison among classical filters, DAE and VAE.
- Quantitative evaluation where ground-truth clean references exist.
- Discussion of generative model limitations and hallucination.

## Future Scope
High-resolution paired datasets, U-Net restoration, perceptual losses, super-resolution, scratch removal, diffusion fine-tuning, GPU deployment, downloadable PDF reports and cloud hosting.
