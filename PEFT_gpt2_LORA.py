''' 
Parameter efficient fine-tuning with LoRA for GPT-2 using PEFT library
This script demonstrates how to apply Low-Rank Adaptation (LoRA) to fine-tune a GPT-2 model
using the PEFT library. It includes loading the model, applying LoRA, and saving the fine-tuned model.

Installation:
    python -m venv peft_venv
    source peft_venv/bin/activate  # On Windows use `peft_venv\Scripts\activate`
    pip install -r requirements_peft.txt

    Alternative installation:
        python3.9 -m venv peft_test_env
        source peft_test_env/bin/activate
        pip install --upgrade pip
        pip install tensorflow==2.10.0
        python -c "import tensorflow as tf; print('tf',tf.__version__)"
    
Acknowledgements:
     https://keras.io/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora/

Usage:
    python PEFT_gpt2_LORA.py

Author: Soumya Banerjee
'''

# Import necessary libraries. Guard heavy native imports so the module can be
# imported without immediately crashing on machines where TensorFlow/other
# native wheels are incompatible.
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # best-effort; ignored if TF not present

import time
import matplotlib.pyplot as plt

# Try importing TensorFlow and datasets only when available. If the import
# fails (for example due to an incompatible native wheel causing 'illegal
# hardware instruction'), record the exception and continue so the file can
# be inspected/edited without crashing the interpreter on import.
_TF_AVAILABLE = False
_TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow import keras
    # set mixed precision policy if supported
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        # ignore if mixed precision isn't available on this platform
        pass
    _TF_AVAILABLE = True
except Exception as e:
    _TF_IMPORT_ERROR = e


def check_tf():
    """Return (available: bool, error: Exception|None)."""
    return _TF_AVAILABLE, _TF_IMPORT_ERROR


def load_reddit_ds(split="train", as_supervised=True):
    """Load the reddit_tifu dataset using tensorflow_datasets.

    Raises a helpful RuntimeError if TensorFlow/tfds failed to import.
    """
    available, err = check_tf()
    if not available:
        raise RuntimeError(
            "TensorFlow or tensorflow_datasets failed to import. Original error: {}".format(err)
        )
    return tfds.load("reddit_tifu", split=split, as_supervised=as_supervised)


if __name__ == "__main__":
    # When run as a script, attempt to load the dataset and report status.
    available, err = check_tf()
    if not available:
        print("TensorFlow import failed when running this script.\nError:", err)
        print("Common fixes: use a conda-forge build of TensorFlow, or create a fresh conda env with a compatible wheel for your macOS/CPU.")
    else:
        print("TensorFlow imported OK. Loading dataset...")
        reddit_ds = load_reddit_ds()
        print("Dataset loaded. Example split size may be large — handle accordingly.")

