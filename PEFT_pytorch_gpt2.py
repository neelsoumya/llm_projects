'''
PEFT for GPT-2 model using PyTorch.

Installation:
    python -m venv peft_pytorch_venv
    source peft_pytorch_venv/bin/activate  # On Windows use `peft_pytorch_venv\Scripts\activate`
    pip install -r requirements_peft_pytorch.txt

Tutorial on PEFT and LoRA:
    https://huggingface.co/blog/peft
    https://github.com/neelsoumya/intro_to_LMMs/blob/main/PEFT.md

Usage:
    python PEFT_pytorch_gpt2.py

Author: Soumya Banerjee

'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import matplotlib.pyplot as plt
import time

# check device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f" Using device: {device} \n")

# Load tokenizer and model
str_model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(str_model_name)
tokenizer.pad_token = tokenizer.eos_token # GPT-2 does not have a pad token, so we use eos_token instead

model = AutoModelForCausalLM.from_pretrained(
    str_model_name,
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
)

print("\n Model loaded. \n")
print(" Model name: ", str_model_name)
print(" Model size: {:.2f}M parameters \n", model.num_parameters() / 1_000_000)
print(" Model size (MB): {:.2f} \n", model.num_parameters() * model.dtype.itemsize / (1024 * 1024))

# Configure LoRA
print("\n Configuring LoRA .... \n")
lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM, # for GPT-2
    r = 8, # rank
    lora_alpha = 32, # alpha
    lora_dropout = 0.1, # dropout
    target_modules = ["c_attn", "c_proj"] # target modules to apply LoRA
)

# LoRA Alpha Parameter Explanation

# In this LoRA (Low-Rank Adaptation) configuration, **`lora_alpha = 32`** is a scaling parameter that controls the magnitude of the LoRA updates applied to the model.

## What it does

#The actual scaling factor applied to the LoRA weights is calculated as:

#```
#scaling_factor = lora_alpha / r
#```

#In your case: `32 / 8 = 4`

#This means the LoRA updates will be scaled by a factor of 4 before being added to the original model weights.

## Why it matters

#- **Higher `lora_alpha`**: Stronger influence of the LoRA adaptation on the model (more aggressive fine-tuning)
#- **Lower `lora_alpha`**: Weaker influence (more conservative fine-tuning)

#The ratio `lora_alpha / r` is often kept constant across different rank values to maintain consistent learning dynamics. A common convention is to set `lora_alpha = 2 * r`, though your configuration uses `lora_alpha = 4 * r`, which will result in stronger LoRA updates.

## In practice

#Think of `lora_alpha` as a learning rate multiplier for the LoRA weights. It's a hyperparameter you can tune based on your task:

#- If your fine-tuning is too aggressive or unstable, you might **lower** it
#- If the model isn't adapting enough, you might **increase** it

## Your Configuration

model = get_peft_model(model, lora_config) # apply LoRA to the model
print("\n LoRA configured. \n")

# Print trainable parameters
print(" \n Trainable parameters after performing LoRA \n")
model.print_trainable_parameters() # print trainable parameters

# Load dataset
print("\n Loading dataset \n")
try:
    dataset = load_dataset("reddit_tifu", "short", split="train[:1000]") # load a small subset for quick testing
    print(" \n Dataset loaded \n")
    print("\n Length of dataset:", len(dataset))
except Exception as e:
    print("\n Failed to load reddit data\n")
    print("\n Loading wiki \n")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split = "train[:1000]")


print(" \n Dataset loaded \n") 

