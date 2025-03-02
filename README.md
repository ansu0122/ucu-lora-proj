# Efficient ViT Fine-tuning with LoRA

HuggingFace
https://huggingface.co/ansu0122/vit-lora/tree/main

WandB
https://wandb.ai/andriy-suh-private/lora-project

Fine-tuned CLIP Vit-Base/16 with LoRA.

## Repository Folder Structure

- **lora_clip/**: the module with LoRA implementation for CLIP ViT models.
- **encoder_utils.py**: the utilities for training and inference of the CLIP ViT encoder.
- **gym.py**: the implementation of the training loop with validation, metric logging to WAndB and a checkpoint persistence in HuggingFace.
- **lora_experiments.ipynb**: Training and inference experiments conducted.
- **install.sh**: the script to set up the project environment.