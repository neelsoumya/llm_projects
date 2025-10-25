''' 
Parameter efficient fine-tuning with LoRA for GPT-2 using PEFT library
This script demonstrates how to apply Low-Rank Adaptation (LoRA) to fine-tune a GPT-2 model
using the PEFT library. It includes loading the model, applying LoRA, and saving the fine-tuned model.

Installation:
    python -m venv peft_venv
    source peft_venv/bin/activate  # On Windows use `peft_venv\Scripts\activate`
    pip install -r requirements_peft.txt

Acknowledgements:
     https://keras.io/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora/

Usage:
    python PEFT_gpt2_LORA.py

Author: Soumya Banerjee
'''

# Import necessary libraries
import os
os.environ["KERAS_BACKEND"] = "tensorflow" # Set Keras backend to TensorFlow
import keras_hub
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensforflow_datasets as tfds
import time

keras.mixed_precision.set_global_policy("mixed_float16")

# Load the dataset
reddit_ds = tfds.load("reddit_tifu", 
                      split = "train",
                      as_supervised = True
                      )

