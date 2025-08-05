import torch
from PIL import Image
from tqdm import tqdm
import random
import diffusers # Assumi di avere la libreria diffusers installata

device = "cuda"



@torch.no_grad()
def diffusion_step(latent, t, embedding_cond, embedding_uncond, guidance_scale):
    latent_model_input = scheduler.scale_model_input(latent, t)

    noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=embedding_uncond).sample
    noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_cond).sample

    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    latent = scheduler.step(noise_pred, t, latent).prev_sample
    return latent


@torch.no_grad()
def prompt_to_prompt(source_prompt="", prompt_edit="", guidance_scale=7.5, steps=100, seed=None, width=512, height=512):
    global attention_maps
    width = width - width % 64
    height = height - height % 64
    
    if seed is None: seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)
    
    scheduler.set_timesteps(steps)
    t_start = 0

    init_latent = torch.zeros((1, unet.config.in_channels, height // 8, width // 8), device=device)
    noise = torch.randn(init_latent.shape, generator=generator, device=device)
    zt = scheduler.add_noise(init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=device)).to(device)

    with torch.amp.autocast(device):
        # Encode the unconditioned (empty) prompt for classifier-free guidance
        uncondtioned_prompt = ""
        tokens_unconditional = clip_tokenizer(uncondtioned_prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

        # Encode the actual (conditional) prompt to guide image generation
        tokens_conditional = clip_tokenizer(source_prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state
        
        # #Process prompt editing
        # if prompt_edit is not None:
        #     tokens_conditional_edit = clip_tokenizer(prompt_edit, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        #     embedding_conditional_edit = clip(tokens_conditional_edit.input_ids.to(device)).last_hidden_state
            

        timesteps = scheduler.timesteps[t_start:]
        
        # === DIFFUSION STEP ===
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            # Chiamo la funzione modificata
            zt = diffusion_step(zt, t, embedding_conditional, embedding_unconditional, guidance_scale)


        z0 = zt / 0.18215
        source_image = vae.decode(z0.to(vae.dtype)).sample

    source_image = (source_image / 2 + 0.5).clamp(0, 1)
    source_image = source_image.cpu().permute(0, 2, 3, 1).numpy()
    source_image = (source_image[0] * 255).round().astype("uint8")
    source_image = Image.fromarray(source_image) 

    edited_image = 1 
    return source_image, edited_image