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

# Preprocess dataset
def tokenize_function(examples):
    ''' Function to tokenize 
        examples: dataset examples
        return: tokenized examples
    '''

    if "documents" in examples:
        texts = examples["documents"] # reddit_tifu
    elif "text" in examples:
        texts = examples["text"] # wikitext
    else:
        texts = [str(ex) for ex in examples if ex] # fallback

    return tokenizer(texts, # tokenize the texts
                     truncation = True, # truncate to max length
                     max_length = 128, # max length
                     padding = "max_length" # pad to max length
                     )

# tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, # tokenize the dataset
                                 batched = True, # batch processing
                                 remove_columns = dataset.column_names # remove original columns
                                 )
print("\n Dataset tokenized. \n")

# Add labels for causal language modeling

# Explanation: Preparing the dataset so the Trainer / model can compute the causal-language-model loss. Concretely:

#* `tokenized_dataset.map(...)` runs a function over each batch (because `batched=True`) of tokenized examples.
#* The lambda `{"labels": examples["input_ids"]}` creates a new column called `labels` whose values are the same as the `input_ids` for each example.
#* Hugging Face `Trainer` and the model expect a `labels` tensor for computing the loss. For causal LMs you give the model the input token ids as the targets too — the model internally shifts logits so position *t* is compared to the token at *t+1*, which implements next-token prediction.

#Two important practical notes / improvements:

#1. **Padding tokens should be ignored in the loss.**
#   We used `padding="max_length"`, so the `input_ids` include pad tokens. The loss function treats labels equal to `-100` as “ignore this token”, so you should replace pad token ids in `labels` with `-100` to avoid penalising the model for padded positions.

# 2. **The mapping is batched, so `examples["input_ids"]` is a list (or tensor) of sequences.** The map returns the same structure but with an added `labels` field.

# A safer version that masks padding would be:

#```python
#def add_labels(examples):
#    labels = examples["input_ids"].copy()   # batched lists
#    # replace pad token id with -100 so loss ignores padded positions
#    pad_id = tokenizer.pad_token_id
#    for i, seq in enumerate(labels):
#        labels[i] = [tok if tok != pad_id else -100 for tok in seq]
#    return {"labels": labels}

#tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
#```

#In short: that code makes the dataset include `labels = input_ids` so the causal LM can compute next-token loss; just be sure to mask padding with `-100` if you used padding.

tokenized_dataset = tokenized_datasets.map(
    lambda examples: { "labels": examples["input_ids"] }, # add labels
    batched = True # batch processing
)

print("\n Labels added to dataset. \n")

# Training arguments
training_args = TrainingArguments(
    output_dir = "./gpt2_lora_output", # output directory
    num_train_epochs = 2, # number of training epochs
    per_device_train_batch_size = 4, # batch size per device
    gradient_accumulation_steps = 4, # gradient accumulation steps
    learning_rate = 3e-4, # learning rate
    logging_steps = 10, # logging steps
    save_steps = 50, # save steps
    fp16 = device == "cuda", # only use fp16 on CUDA
    report_to = "none" # no reporting to wandb/tensorboard
)

# TODO: setup reporting to wandb/tensorbaord


