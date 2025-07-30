# InstructPix2Pix implementation

## Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:
```bash 
pip install -r requirements.txt
```
and initialize an [accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

## Training InstructPix2Pix

Training a model like InstructPix2Pix can be demanding on your hardware. However, by enabling `gradient_checkpointing` and `mixed_precision`, it's possible to train on a **single 24GB GPU**.
For faster training or larger batch sizes, we **strongly recommend** using GPUs with **at least 30GB of memory**.

We'll train using a [small toy dataset](https://huggingface.co/datasets/fusing/instructpix2pix-1000-samples), which is a downsized version of the [original dataset](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) from the InstructPix2Pix paper.

Before launching training, configure the model and dataset:

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATASET_ID="fusing/instructpix2pix-1000-samples"
```

Use `accelerate` to start the training process with the following configuration:

```bash
accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --resolution=256 --random_flip \
    --train_batch_size=32 --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --max_train_steps=100 \
    --checkpointing_steps=50 --checkpoints_total_limit=2 \
    --learning_rate=5e-5 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --push_to_hub
```

> These settings have been successfully tested on a **40GB NVIDIA A100 GPU**.


## Edit a single image

To edit an image using the `test_instruct_pix2pix.py` script, first configure the model and the URL of the image to edit:

```bash
export MODEL_PATH="iamlucaconti/instruct-pix2pix-model"
export URL="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png"
````

Then run the following command in your terminal:

```bash
python test_instruct_pix2pix.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --image_url $URL \
  --prompt "make it a van gogh painting" \
  --seed 42 \
  --num_inference_steps 100 \
  --image_guidance_scale 2.0 \
  --guidance_scale 8.0 \
  --generated_images_path "./results"
```

Make sure you have the required dependencies installed and that your environment (e.g., Python, PyTorch, Hugging Face libraries) is properly set up.
The output image will be saved in the `./results` folder.

## References

- Brooks, Tim, Aleksander Holynski, and Alexei A. Efros.  **"InstructPix2Pix: Learning to Follow Image Editing Instructions."** arXiv preprint [arXiv:2211.09800](https://arxiv.org/abs/2211.09800), 2022.

