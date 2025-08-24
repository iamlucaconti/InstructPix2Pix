
# Image Editing with Diffusion Models


# !!STILL WORK IN PROGRESS!!

## Authors
*   [**Luca Conti**](https://github.com/iamlucaconti/) (ID: 1702084)
*   [**Daniele sabatini**](https://github.com/danilsab24/) (ID: 1890300)


## Project aim and selected paper
This project is based on the implementation of the paper [**InstructPix2Pix: Learning to Follow Image Editing Instructions**](https://arxiv.org/abs/2211.09800).  

Our implementation is divided into three main parts:

1. **Fine-tuning LLM**  
   In the first part, we fine-tuned a LLM model to create a dataset containing original prompts, edit prompt and edited prompts that would later be used for the second component of the project.

2. **Image Generation with Stable Diffusion and ControlNet**  
   In the second part, we used the prompts generated in the first part to create pairs of images, where one was the edited version of the other (i.e., maintaining coherence between them).  
   For this, we employed **Stable Diffusion** together with **ControlNet**.  
   The generated images were then used to build the dataset required for the final part of the project.

3. **Fine-tuning InstructPix2Pix**  
   In the last part of the project, we used the dataset produced in the second part to fine-tune a Stable Diffusion model.


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

## 1. Generating Instructions and Paired Captions


>To finetune a LLM to generate a dataset of paired captions, run the notebook **`generate_txt_dataset.ipynb`** and edit the variables in the corresponding section.

The notebook **`generate_txt_dataset.ipynb`** implements the first stage of the project.  
It contains the code for fine-tuning the [**Gemma 2B**](https://huggingface.co/google/gemma-2b) model to generate an **`edit_prompt`** and an **`edited_prompt`** given an **`original_prompt`**.

For fine-tuning, we built a dataset of **800 samples**:

- **400 manually created samples**:  Starting from the `original_prompt` in the `TEXT` column of the [**laion/aesthetics_v2_4.75**](https://huggingface.co/datasets/laion/aesthetics_v2_4.75) dataset, we manually designed both the `edit_prompt` and the `edited_prompt`.  

- **400 samples from existing data**:  
  Taken directly from the [**InstructPix2Pix training dataset**](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered), as used in the original paper.


The dataset used for fine-tuning Gemma 2B is available at:
 ```
 ./datasets/txt_dataset_finetuning.jsonl
 ```

Using the fine-tuned LLM, we then prepared the paired caption dataset for the second stage of the project:

- **2,900 samples** from [**visual-layer/imagenet-1k-vl-enriched**](https://huggingface.co/datasets/visual-layer/imagenet-1k-vl-enriched).  
  For each sample, the `original_prompt` was passed to the fine-tuned model, which generated the corresponding `edit_prompt` and `edited_prompt`.  

- **150 manually created samples**:  
  Added to further enrich the dataset.


We provide our generated dataset of captions and edit instructions in:
 ```
 ./datasets/generated_txt_dataset.jsonl
 ```

## 2. Generating Paired Images from Paired Captions

>To generate a dataset of paired images from paired captions, run the notebook **`generate_img_dataset.ipynb`** and edit the variables in the corresponding section.

To generate the original images, we used the Stable Diffusion model `runwayml/stable-diffusion-v1-5`.  
Image editing is guided by `lllyasviel/sd-controlnet-canny`, which uses the source image structure (Canny edges) along with the target prompt to produce controlled edits.  
>**Note**  
[Prompt-to-Prompt](https://arxiv.org/abs/2208.01626) replaces *cross-attention* weights in the second generated image differently depending on the type of edit (e.g., word swap, adding a phrase, increasing or decreasing the weight of a word).  
[Brooks et al.](https://arxiv.org/abs/2211.09800), on the other hand, replaced *self-attention* weights of the second image during the first $p$ fraction of diffusion steps and applied the same attention weight replacement strategy for all edits.  
However, replacing self-attention weights is computationally expensive. Moreover, finding the optimal combination of parameters for the original Prompt-to-Prompt method requires extensive search for each image. For these reasons, we opted to use [ControlNet](https://arxiv.org/abs/2302.05543) (Canny) instead. The only parameter we performed a search over was `controlnet_conditioning_scale`.

All images are generated at a resolution of `HEIGHT = 512 x WIDTH = 512` using `STEPS = 40` diffusion steps. A fixed seed (`SEED1=20230330`) ensures reproducibility.  
To evaluate the generated images and enable filtering, we used the CLIP model `openai/clip-vit-large-patch14` to measure semantic and visual similarity between images and text.  
The figure below shows the result of generating an image with the **`original_prompt`** (left) and the **`edited_prompt`** (right), without using ControlNet guidance.

![Without ControlNet](images/without_controlnet.png)  

The two images are not consistent (e.g., the bear is in a different position and the overall composition changes).  
By using ControlNet, we obtain the following result, where the structure of the source image is preserved:

![With ControlNet](images/with_controlnet.png)




The `controlnet_best` function generates a source image with Stable Diffusion and then creates a target image using ControlNet guided by Canny edges. Key steps include:

1. Generate the source image (`image_sd`) from `src_prompt`.
2. Preprocess the image for ControlNet:

   * Convert to Canny edges.
   * Resize to target resolution.
3. Iterate over multiple `controlnet_conditioning_scale` values to generate candidate edited images.
4. Evaluate candidate images using:

   * CLIP directional similarity (`dir_sim`)
   * Image-to-image similarity (`img_img_sim`)
   * Image-to-caption similarity (`img_cap_sim`)
5. Filter images based on thresholds (`dir_sim >= 0.15`, `img_img_sim >= 0.65`, `img_cap_sim >= 0.2`).
6. Return the image with the highest directional similarity.

>**Note**  
Our generated dataset of 1001 samples is available on Hugging Face ([here](https://huggingface.co/datasets/iamlucaconti/instructpix2pix-controlnet)).




## 3. Training InstructPix2Pix

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


## Limitations and reflections

### Generated Instructions and Paired Captions
* **Hardware constraints**:  Since the experiments were conducted on **Google Colab Pro**, which provides only **40 GB of A100 GPU memory**, it was not possible to fine-tune larger models such as **GEMMA 7B** or **GPT-3**(like in the original paper).  
This restriction limited the scalability of our experiments and made the final dataset less precise compared to the one presented in the original paper.

* **Dataset constraints**: The dataset used for fine-tuning was **partially created manually**, which resulted in a relatively **small dataset size**.  
This limitation affects the quality of the fine-tuning process, reducing the model’s ability to generalize effectively.

### Generated Paired Images from Paired Captions

* **Dependence on Pretrained Models:** The quality and diversity of the generated dataset are bounded by the pretrained model’s biases, i.e., if SD struggles with certain subjects or styles, the dataset will inherit those limitations.

* **Synthetic Data Bias**: Both original and edited images are synthetically generated, not real-world images. This can lead to a domain gap when fine-tuning InstructPix2Pix for real-world tasks, reducing generalization.

* **Prompt Quality and Variability**: The dataset depends heavily on the quality and specificity of the original and edited prompt (the latter are generated by a finetuned LLM).


* **ControlNet Conditioning Scale**: We experimented with a limited range of `controlnet_conditioning_scale` values and filtered the generated images using similarity metrics. A broader range of `controlnet_conditioning_scale` choices would likely improve coverage and diversity.  
Also, selecting thresholds (`dir_sim >= 0.15`, `img_img_sim >= 0.65`, `img_cap_sim >= 0.2`) is heuristic and may bias the dataset toward safe or easy edits, ignoring harder transformations.


* **Similarity Metrics Limitations**: We relied on CLIP-based directional similarity, image-image similarity, and image-caption similarity. CLIP may fail for subtle edits or abstract concepts, meaning some high-quality pairs could be discarded. Metrics are not perfect proxies for human perception; some images passing thresholds may still be semantically incorrect (and viceversa).

## References

* Tim Brooks, Aleksander Holynski, Alexei A. Efros.
  *InstructPix2Pix: Learning to Follow Image Editing Instructions.*
  arXiv:2211.09800, 2022.
  [https://arxiv.org/abs/2211.09800](https://arxiv.org/abs/2211.09800)

* Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, Daniel Cohen-Or.
  *Prompt-to-Prompt Image Editing with Cross Attention Control.*
  arXiv:2208.01626, 2022.
  [https://arxiv.org/abs/2208.01626](https://arxiv.org/abs/2208.01626)

* Lvmin Zhang, Maneesh Agrawala.
  *Adding Conditional Control to Text-to-Image Diffusion Models.*
  arXiv:2302.05543, 2023.
  [https://arxiv.org/abs/2302.05543](https://arxiv.org/abs/2302.05543)

* Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer.  
  [*High-Resolution Image Synthesis with Latent Diffusion Models.*](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
  CVPR 2022.

