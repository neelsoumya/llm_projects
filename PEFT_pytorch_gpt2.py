'''
PEFT for GPT-2 model using PyTorch.

Installation:
    python -m venv peft_pytorch_venv
    source peft_pytorch_venv/bin/activate  # On Windows use `peft_pytorch_venv\Scripts\activate`
    pip install -r requirements_peft_pytorch.txt

'''

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
