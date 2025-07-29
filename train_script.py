from datasets import load_dataset
dataset = load_dataset("fusing/instructpix2pix")
# Il dataset è tipicamente diviso in 'train' e 'validation'
train_dataset = dataset["train"]


import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL # <-- AGGIUNTA QUI
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os

# 1. Caricare il modello pre-addestrato
# Carichiamo i componenti di Stable Diffusion v1.5
# Se hai problemi con l'autenticazione di Hugging Face, devi fare login nel tuo notebook
# da terminale: !huggingface-cli login
model_id = "runwayml/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

# Per InstructPix2Pix, dovrai modificare l'UNet o creare un modello wrapper
# che accetti l'immagine originale come input aggiuntivo.
# Questo è il punto più complesso. Potresti aver bisogno di modificare la forward pass del UNet
# per includere l'encoding dell'immagine originale.
# Un modo per iniziare è usare un ControlNet o un modello che ha già questa capacità.

# Inizializza lo scheduler
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# 2. Preparare il dataset
# Questo è un esempio generico, dovrai adattarlo al tuo dataset
class InstructPix2PixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        original_image = item["image"] # Assumi che il dataset abbia una chiave 'image' per l'originale
        edited_image = item["edited_image"] # E una chiave 'edited_image' per l'output desiderato
        prompt = item["edit_prompt"] # E una chiave 'edit_prompt' per l'istruzione di testo

        # Applica le trasformazioni
        if self.transform:
            original_image = self.transform(original_image.convert("RGB"))
            edited_image = self.transform(edited_image.convert("RGB"))

        # Tokenizza il prompt
        input_ids = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "original_pixel_values": original_image,
            "edited_pixel_values": edited_image,
            "input_ids": input_ids,
        }

# Trasformazioni per le immagini
image_transforms = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# Carica il dataset (sostituisci con il tuo caricamento del dataset di InstructPix2Pix)
# dataset = load_dataset("fusing/instructpix2pix") # Esempio di caricamento
# train_dataset = InstructPix2PixDataset(dataset["train"], transform=image_transforms)
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 3. Configurare l'acceleratore per l'addestramento distribuito
accelerator = Accelerator(
    gradient_accumulation_steps=1, # O più alto se la memoria è un problema
    mixed_precision="fp16", # o "bf16" se la GPU lo supporta
)

# Sposta i modelli sul dispositivo corretto e prepara per l'addestramento
unet, text_encoder, vae = accelerator.prepare(unet, text_encoder, vae)
# Se usi un ControlNet, aggiungilo qui: controlnet = accelerator.prepare(controlnet)

# 4. Ottimizzatore
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5) # Fine-tuneremo principalmente l'UNet
optimizer = accelerator.prepare(optimizer)

# 5. Loop di addestramento
num_train_epochs = 10
for epoch in range(num_train_epochs):
    unet.train()
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            # Pre-elaborazione delle immagini (encoding VAE)
            latents = vae.encode(batch["edited_pixel_values"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Generazione del rumore per la diffusione
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()

            # Aggiungi rumore ai latenti
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Condizionamento del testo
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # In questa parte dovresti includere l'encoding dell'immagine originale
            # Questo è il punto in cui l'architettura di InstructPix2Pix differisce.
            # Se stai usando un ControlNet, passeresti l'immagine originale a ControlNet
            # e useresti il suo output per condizionare l'UNet.
            # Se stai replicando InstructPix2Pix, la tua UNet modificata accetterebbe
            # un input addizionale per l'immagine originale.
            # Esempio concettuale (da adattare):
            # original_image_embeddings = unet.encode_image_input(batch["original_pixel_values"])
            # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, original_image_embeddings).sample

            # Per ora, usiamo la versione standard di Stable Diffusion per mostrare il loop
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Calcola la loss (MSE tra il rumore predetto e il rumore effettivo)
            # InstructPix2Pix usa una loss combinata, inclusa una loss di percettivo (LPIPS)
            # e una loss su un "clip-guidance". Per iniziare, puoi usare MSE.
            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            # Backward pass e ottimizzazione
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # Stampa lo stato dell'addestramento
        if step % 100 == 0:
            accelerator.print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Salva il modello alla fine di ogni epoch (o periodicamente)
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(f"unet_instructpix2pix_epoch_{epoch}")