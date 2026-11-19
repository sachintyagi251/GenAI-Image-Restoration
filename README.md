# GenAI Image Denoising & Restoration

A B.Tech Generative AI project using a Denoising Autoencoder (DAE), Variational Autoencoder (VAE), classical filters, optional diffusion img2img, and Streamlit.

## Setup (Windows)

1. Install Python 3.10 or 3.11 recommended.
2. Open this folder in VS Code.
3. Open Terminal (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Train the models

CIFAR-10 is downloaded automatically. Images are resized to 64x64 and Gaussian noise is added during training.

Train the Denoising Autoencoder:
```powershell
python train.py --model dae --epochs 10
```

Train the VAE:
```powershell
python train.py --model vae --epochs 10
```

Train both:
```powershell
python train.py --model both --epochs 10
```

For a quick smoke test, try `--epochs 1`. Training can be slow on CPU. If available, CUDA GPU is recommended. Checkpoints are saved in `checkpoints/`.

## Run the web app

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Available restoration modes

- Gaussian filter
- Median filter
- Bilateral filter
- Denoising Autoencoder (requires `checkpoints/dae.pth`)
- VAE reconstruction (requires `checkpoints/vae.pth`)
- Optional diffusion img2img (requires extra dependencies/model download)

## Optional diffusion installation

Diffusion is optional and requires a compatible Hugging Face Diffusers environment, internet access for downloading weights, and preferably a CUDA GPU. Install:

```powershell
pip install diffusers transformers accelerate safetensors
```

Some model repositories may require accepting a model license and authenticating with Hugging Face. The app's img2img diffusion mode may alter or invent image details; do not treat it as faithful restoration.

## Metrics

PSNR, SSIM and MSE are computed only when a clean reference image is supplied. A valid comparison needs aligned images of the same scene. Comparing output to the degraded input is not a true restoration metric.

## Project limitations

- CIFAR-10 consists of low-resolution 32x32 images; resized to 64x64 for training does not add real detail.
- Trained models may not generalize well to high-resolution portraits, documents, or old photographs.
- Diffusion output is generative and may modify identities, text, objects, or details.
- This is a research/academic prototype, not a professional archival restoration system.

## Suggested next upgrades

- Add a paired high-resolution dataset such as DIV2K or BSD300.
- Train a U-Net with perceptual loss.
- Add scratch/blur degradation synthesis.
- Add restoration history, user authentication, report export and model-comparison charts.
- Add an experiment page that plots training loss and compares PSNR/SSIM.
