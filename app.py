import io, os
import streamlit as st
import numpy as np
import torch
from PIL import Image, ImageFilter
import cv2
from skimage.metrics import structural_similarity
from models import DenoisingAutoencoder, VAE

st.set_page_config(page_title="GenAI Image Restoration", page_icon="🖼️", layout="wide")
st.title("🖼️ GenAI Image Denoising & Restoration")
st.write("Restore noisy, blurred and degraded images using classical filters and trained deep-learning models.")

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"Running on: **{DEVICE}**")
method=st.sidebar.selectbox("Restoration method",[
    "Gaussian Filter","Median Filter","Bilateral Filter","Denoising Autoencoder (trained)","VAE Reconstruction (trained)","Diffusion (optional)"
])
noise_level=st.sidebar.slider("Noise strength (for demo corruption)",0.0,0.5,0.15,0.01)
st.sidebar.caption("Noise strength is used when you choose to add synthetic noise below.")

def pil_to_tensor(image, size=64):
    arr=np.asarray(image.convert("RGB").resize((size,size)),dtype=np.float32)/255.
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

def tensor_to_pil(t):
    arr=t.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    arr=np.clip(arr*255,0,255).astype(np.uint8)
    return Image.fromarray(arr)

@st.cache_resource
def load_dae(path):
    model=DenoisingAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))
    model.eval(); return model

@st.cache_resource
def load_vae(path):
    model=VAE().to(DEVICE)
    model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))
    model.eval(); return model

def process_image(img, selected):
    if selected=="Gaussian Filter":
        return img.filter(ImageFilter.GaussianBlur(radius=1.2))
    if selected=="Median Filter":
        return img.filter(ImageFilter.MedianFilter(size=3))
    if selected=="Bilateral Filter":
        arr=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
        out=cv2.bilateralFilter(arr,9,75,75)
        return Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
    if selected.startswith("Denoising Autoencoder"):
        path="checkpoints/dae.pth"
        if not os.path.exists(path): raise FileNotFoundError("Checkpoint missing. Train the DAE using `python train.py --model dae`.")
        model=load_dae(path)
        x=pil_to_tensor(img).to(DEVICE)
        with torch.no_grad(): out=model(x)
        return tensor_to_pil(out).resize(img.size,Image.Resampling.LANCZOS)
    if selected.startswith("VAE"):
        path="checkpoints/vae.pth"
        if not os.path.exists(path): raise FileNotFoundError("Checkpoint missing. Train VAE using `python train.py --model vae`.")
        model=load_vae(path)
        x=pil_to_tensor(img).to(DEVICE)
        with torch.no_grad(): out,_,_=model(x)
        return tensor_to_pil(out).resize(img.size,Image.Resampling.LANCZOS)
    if selected=="Diffusion (optional)":
        prompt=st.session_state.get("diff_prompt","Restore this image with realistic details, preserve its content.")
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            model_id=os.getenv("SD_MODEL_ID","runwayml/stable-diffusion-v1-5")
            pipe=StableDiffusionImg2ImgPipeline.from_pretrained(model_id,torch_dtype=torch.float16 if DEVICE.type=="cuda" else torch.float32)
            pipe=pipe.to(DEVICE)
            if DEVICE.type=="cuda": pipe.enable_attention_slicing()
            result=pipe(prompt=prompt,image=img.resize((512,512)),strength=0.35,guidance_scale=7.0,num_inference_steps=25).images[0]
            del pipe
            return result.resize(img.size,Image.Resampling.LANCZOS)
        except Exception as e:
            raise RuntimeError(f"Diffusion setup failed: {e}. Install diffusers and use a GPU where possible.")
    return img

def calculate_metrics(reference, result):
    a=np.asarray(reference.convert("RGB").resize(result.size),dtype=np.uint8)
    b=np.asarray(result.convert("RGB"),dtype=np.uint8)
    mse=float(np.mean((a.astype(float)-b.astype(float))**2))
    psnr=float("inf") if mse==0 else 10*np.log10((255**2)/mse)
    ssim=float(structural_similarity(a,b,channel_axis=2,data_range=255))
    return mse,psnr,ssim

uploaded=st.file_uploader("Upload an image",type=["png","jpg","jpeg","webp"])
if uploaded:
    original=Image.open(uploaded).convert("RGB")
    st.subheader("Input image")
    st.image(original,caption=f"Uploaded image • {original.width} × {original.height}",use_container_width=True)
    col1,col2=st.columns(2)
    with col1:
        add_noise=st.checkbox("Add synthetic Gaussian noise to input")
    with col2:
        reference_file=st.file_uploader("Optional: upload clean reference image for metrics",type=["png","jpg","jpeg","webp"],key="ref")
    working=original.copy()
    if add_noise:
        arr=np.asarray(original,dtype=np.float32)
        arr=np.clip(arr+np.random.normal(0,noise_level*255,arr.shape),0,255).astype(np.uint8)
        working=Image.fromarray(arr)
        st.image(working,caption="Synthetic noisy input",use_container_width=True)
    if method=="Diffusion (optional)":
        st.text_input("Prompt for restoration",key="diff_prompt",value="Restore this image, remove noise and scratches, preserve original objects and identity, natural details.")
        st.warning("Diffusion can invent or alter details. It is not guaranteed to preserve the original image exactly.")
    if st.button("✨ Restore Image",type="primary",use_container_width=True):
        with st.spinner("Processing image..."):
            try:
                restored=process_image(working,method)
                st.session_state["restored"]=restored
                st.session_state["working"]=working
                st.session_state["original"]=original
                st.session_state["method"]=method
            except Exception as e:
                st.error(str(e))
    if "restored" in st.session_state:
        restored=st.session_state["restored"]
        st.subheader("Restoration results")
        left,right=st.columns(2)
        with left: st.image(st.session_state["working"],caption="Before restoration",use_container_width=True)
        with right: st.image(restored,caption="After restoration",use_container_width=True)
        st.write(f"Method: **{st.session_state['method']}**")
        if reference_file:
            reference=Image.open(reference_file).convert("RGB")
            mse,psnr,ssim=calculate_metrics(reference,restored)
            st.subheader("Quality metrics (against clean reference)")
            m1,m2,m3=st.columns(3)
            m1.metric("MSE",f"{mse:.3f}")
            m2.metric("PSNR",f"{psnr:.2f} dB")
            m3.metric("SSIM",f"{ssim:.4f}")
            st.caption("These metrics are meaningful when the reference image is aligned with the restored result.")
        else:
            st.info("Upload an aligned clean reference image to calculate PSNR, SSIM and MSE. Without a reference, these metrics cannot objectively measure restoration quality.")
        buf=io.BytesIO()
        restored.save(buf,format="PNG")
        st.download_button("⬇️ Download restored image",buf.getvalue(),"restored_image.png","image/png",use_container_width=True)
else:
    st.info("Upload an image above to start.")
st.divider()
st.caption("Academic prototype. Trained models are trained on CIFAR-10 resized to 64×64; performance on arbitrary photographs may vary. Diffusion may change image content.")
