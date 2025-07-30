import argparse
import os
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for InstructPix2Pix.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )

    parser.add_argument(
        "--image_url",
        type=str,
        required=True,
        help="URL of the image to be edited."
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired modification of the input image."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility."
    )
    
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="Number of denoising steps during inference."
    )

    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=2.0,
        help="How much the image influences the result (higher means more faithful to image)."
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=8.0,
        help="How much the prompt influences the result (higher means more faithful to prompt)."
    )
    
    parser.add_argument(
        "--generated_images_path",
        type=str,
        default="./generated_images",
        help="Path to directory where edited images will be saved."
    )

    return parser.parse_args()


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.generated_images_path, exist_ok=True)

    # Load model
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    # Set seed
    generator = torch.Generator("cuda").manual_seed(args.seed)

    # Load image
    image = download_image(args.image_url)

    # Generate image
    edited_image = pipe(
        prompt=args.prompt,
        image=image,
        num_inference_steps=args.num_inference_steps,
        image_guidance_scale=args.image_guidance_scale,
        guidance_scale=args.guidance_scale,
        generator=generator
    ).images[0]

    # Create safe filename from prompt
    safe_prompt = "".join(c if c.isalnum() else "_" for c in args.prompt.strip())[:50]
    output_path = os.path.join(args.generated_images_path, f"{safe_prompt}.png")

    # Save image
    edited_image.save(output_path)
    print(f"Image saved at: {output_path}")

    # Optional: show image

    # fontsize = 18

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs[0].imshow(image)
    # axs[0].axis("off")
    # axs[0].set_title("Original Image", fontsize=fontsize)

    # # Mostra original & editato
    # axs[1].imshow(edited_image)
    # axs[1].axis("off")
    # axs[1].set_title(f"'{prompt}'", fontsize=fontsize)

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()

    

