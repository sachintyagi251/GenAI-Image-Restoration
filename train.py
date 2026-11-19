import argparse, os, random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import DenoisingAutoencoder, VAE

def seed_everything(seed=42):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_loaders(batch_size, workers):
    tfm = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=tfm)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers))

def train_dae(model, loader, device, epochs, noise_std, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        total = 0
        for clean, _ in loader:
            clean = clean.to(device)
            noisy = torch.clamp(clean + torch.randn_like(clean)*noise_std, 0, 1)
            restored = model(noisy)
            loss = F.mse_loss(restored, clean)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print(f"DAE Epoch {epoch+1}/{epochs} | loss={total/len(loader):.5f}")
        torch.save(model.state_dict(), save_path)
    print("Saved:", save_path)

def train_vae(model, loader, device, epochs, save_path, beta):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        total = 0
        for clean, _ in loader:
            clean = clean.to(device)
            noisy = torch.clamp(clean + torch.randn_like(clean)*0.10, 0, 1)
            recon, mu, logvar = model(noisy)
            reconstruction = F.mse_loss(recon, clean, reduction="sum") / clean.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / clean.size(0)
            loss = reconstruction + beta*kl
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print(f"VAE Epoch {epoch+1}/{epochs} | loss={total/len(loader):.5f}")
        torch.save(model.state_dict(), save_path)
    print("Saved:", save_path)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dae","vae","both"], default="both")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--beta", type=float, default=0.0005)
    args=parser.parse_args()
    seed_everything()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)
    train_loader, _=make_loaders(args.batch_size,args.workers)
    os.makedirs("checkpoints",exist_ok=True)
    if args.model in ("dae","both"):
        dae=DenoisingAutoencoder().to(device)
        train_dae(dae,train_loader,device,args.epochs,args.noise,"checkpoints/dae.pth")
    if args.model in ("vae","both"):
        vae=VAE().to(device)
        train_vae(vae,train_loader,device,args.epochs,"checkpoints/vae.pth",args.beta)
