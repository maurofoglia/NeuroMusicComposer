"""
Unified EEG Emotion Recognition Pipeline

This script serves as the single entry point for training the emotion recognition
model. It dynamically conditionally executes either the baseline linear pipeline
(e.g., Muse -> LUNA) or the upsampled foundation pipeline (Muse -> Zuna -> BIOT)
based on a single boolean configuration flag.
"""

import os
import torch

from src.braindecode_framework.model_configs import get_model_extra_kwargs
from src.dataset.dataset import Dataset
from src.braindecode_framework.braindecode_framework import BrainDecodeFramework

# IMPORTANTE: Importiamo le funzioni Zuna dal nuovo modulo di utilità
from src.utils.zuna_transforms import convert_dataset_to_biot_bipolar

"""
===============================================================================
1. MASTER CONFIGURATION
===============================================================================
"""

# MASTER SWITCH: Toggle between Baseline and Zuna/BIOT pipelines
USE_ZUNA_PIPELINE = True

# Common Parameters
LOW_CUT_HZ = 1
HIGH_CUT_HZ = 45
TRIAL_DURATION_SEC = 4  # Duration of a single trial in seconds

# Dynamic Setup based on the Master Switch
if USE_ZUNA_PIPELINE:
    MODEL_NAME = "BIOT"
    N_CHANS = 16  # Bipolar derivations
    SFREQ = 256  # Upsampled frequency
    FILE_SUFFIX = "zuna_biot_upsampled"
else:
    MODEL_NAME = "LUNA"
    N_CHANS = 4  # Standard Muse 2 channels
    SFREQ = 200  # Baseline frequency
    FILE_SUFFIX = "baseline"

# Automatically calculate the required input window size
INPUT_WINDOW_SAMPLES = int(TRIAL_DURATION_SEC * SFREQ)

# Dynamic File Paths
DATASET_DIR = 'dataset'
os.makedirs(DATASET_DIR, exist_ok=True)
DATASET_PKL_PATH = os.path.join(DATASET_DIR, f'dataset_{FILE_SUFFIX}.pkl')
PREPROCESSED_PKL_PATH = os.path.join(DATASET_DIR, f'preprocessed_{MODEL_NAME}_{FILE_SUFFIX}.pkl')

print(f"\n[INIT] Starting pipeline: {FILE_SUFFIX.upper()}")
print(f"[INIT] Model: {MODEL_NAME} | Channels: {N_CHANS} | SFreq: {SFREQ}Hz | Window: {INPUT_WINDOW_SAMPLES}")

"""
===============================================================================
2. DATA INGESTION & PIPELINE CONDITIONING
===============================================================================
"""

dataset = Dataset()
dataset.load_user_path(path_in='BIDS_clean')  # o BIDS_zuna_BIOT a seconda delle tue folder

# .xdf --> .fif (Uncomment if needed)
# dataset.expand_bids_with_fif(bids_root='BIDS_clean')

# Conditionally load data and apply transformations
if USE_ZUNA_PIPELINE:
    dataset.load_emotive_raw_data_fif(plot=False, use_zuna=True)

    # Apply Zuna -> BIOT bipolar derivation conversion
    print("\n[PROCESS] Applying Zuna bipolar conversion...")
    convert_dataset_to_biot_bipolar(dataset)

else:
    dataset.load_emotive_raw_data_fif(plot=False, use_zuna=False)

# Save the dataset specific to the active pipeline
dataset.save_pkl_data(path_out=DATASET_PKL_PATH, data=dataset.dataset)

"""
===============================================================================
3. BRAINDECODE PREPROCESSING
===============================================================================
"""

braindecode = BrainDecodeFramework(model_name=MODEL_NAME)

braindecode.load_dataframe(path_in=DATASET_PKL_PATH)
braindecode.create_raw_concat_dataset()
braindecode.preprocess(low_cut_hz=LOW_CUT_HZ, high_cut_hz=HIGH_CUT_HZ)

# Save preprocessed data dynamically
try:
    braindecode.save_pkl_data(
        path_out=PREPROCESSED_PKL_PATH,
        data=braindecode.braindecode_preprocessing,
    )
    braindecode.load_dataframe(path_in=PREPROCESSED_PKL_PATH)
except MemoryError:
    print("\n[WARNING] MemoryError: Continuing in-memory without serializing.")
    braindecode.dataset = braindecode.braindecode_preprocessing

braindecode.extra_model_kwargs = get_model_extra_kwargs(MODEL_NAME, braindecode.dataset)

"""
===============================================================================
4. HARDWARE & DYNAMIC MODEL CREATION
===============================================================================
"""

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"\n[INFO] Allocating model on device: {device.upper()}")

# The model is created using the dynamically calculated parameters
braindecode.create_model(
    n_chans=N_CHANS,
    sfreq=SFREQ,
    input_window_samples=INPUT_WINDOW_SAMPLES,
    n_outputs=2,
    device=device
)

"""
===============================================================================
5. TRAINING EXECUTION
===============================================================================
"""

braindecode.train(
    emotion='excited',
    trial_duration_train=TRIAL_DURATION_SEC,
    epochs_duration_train=4,
    stride_train=4,
    trial_duration_test=TRIAL_DURATION_SEC,
    epochs_duration_test=4,
    stride_test=4
)

print(f"\n[SUCCESS] {FILE_SUFFIX.upper()} pipeline execution completed.")