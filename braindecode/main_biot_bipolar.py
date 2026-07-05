"""
BIOT Bipolar Derivation Pipeline

This script executes the EEG emotion recognition pipeline specifically formatted
for the BIOT architecture. It leverages the Zuna framework for data ingestion,
applies a 16-channel bipolar derivation transformation, and trains the model.
"""

import os
import torch

from src.braindecode_framework.model_configs import get_model_extra_kwargs
from src.dataset.dataset import Dataset
from src.braindecode_framework.braindecode_framework import BrainDecodeFramework

# Import constants and transformation functions from our utility module
from src.utils.zuna_utils import convert_dataset_to_biot_bipolar, drop_dataset_channels

# =============================================================================
# 1. CONFIGURATIONS & PATHS
# =============================================================================

# Signal Processing Parameters
LOW_CUT_HZ = 1
HIGH_CUT_HZ = 45

# Training & Windowing Parameters
SFREQ = 256                 # Upsampled frequency (changed from 128/200 to 256)
TRIAL_DURATION_SEC = 4      # 4 seconds
INPUT_WINDOW_SAMPLES = int(TRIAL_DURATION_SEC * SFREQ)  # Automatically calculates 1024

# Model Selection
MODEL_NAME = "BIOT"         # Available: CTNet, BIOT, REVE, Labram, BENDR

# File Paths (Centralized for easy modification)
BIDS_ROOT_PATH = 'BIDS_zuna_BIOT'
BASE_OUT_DIR = r'E:\Tesi_zuna_reve\standard'

# Ensure the output directory exists
os.makedirs(BASE_OUT_DIR, exist_ok=True)

DATASET_PKL_PATH = os.path.join(BASE_OUT_DIR, 'dataset_bids_zuna_biot_bipolar.pkl')
PREPROCESSED_PKL_PATH = os.path.join(BASE_OUT_DIR, f'braindecode_dataset_preprocessed_{MODEL_NAME}_bipolar.pkl')
SAVE_PREPROCESSED_PKL = True


# =============================================================================
# 2. DATA INGESTION & TRANSFORMATION
# =============================================================================

print("\n[INIT] Loading dataset and applying ZUNA -> BIOT transformations...")

dataset = Dataset()
dataset.load_user_path(path_in=BIDS_ROOT_PATH)

# Optional: Expand BIDS to FIF format
# dataset.expand_bids_with_fif(bids_root='BIDS')

# Load emotive raw data utilizing the ZUNA framework
dataset.load_emotive_raw_data_fif(plot=False, use_zuna=True)

# Convert ZUNA format (20 physical channels) to 16 BIOT bipolar derivations
convert_dataset_to_biot_bipolar(dataset)

# Optional: Drop extraneous channels keeping only ZUNA physicals
# drop_dataset_channels(dataset)

# Save the transformed dataset
dataset.save_pkl_data(path_out=DATASET_PKL_PATH, data=dataset.dataset)


# =============================================================================
# 3. PREPROCESSING
# =============================================================================

print(f"\n[PROCESS] Initializing Braindecode Framework for {MODEL_NAME}...")

braindecode = BrainDecodeFramework(model_name=MODEL_NAME)

braindecode.load_dataframe(path_in=DATASET_PKL_PATH)
braindecode.create_raw_concat_dataset()
braindecode.preprocess(low_cut_hz=LOW_CUT_HZ, high_cut_hz=HIGH_CUT_HZ)

# Attempt to serialize the preprocessed data, fallback to memory if RAM is exhausted
try:
    if SAVE_PREPROCESSED_PKL:
        braindecode.save_pkl_data(
            path_out=PREPROCESSED_PKL_PATH,
            data=braindecode.braindecode_preprocessing,
        )
        braindecode.load_dataframe(path_in=PREPROCESSED_PKL_PATH)
    else:
        braindecode.dataset = braindecode.braindecode_preprocessing
except MemoryError:
    print("[WARNING] MemoryError during serialization: continuing in-memory without saving.")
    braindecode.dataset = braindecode.braindecode_preprocessing

braindecode.extra_model_kwargs = get_model_extra_kwargs(MODEL_NAME, braindecode.dataset)


# =============================================================================
# 4. HARDWARE ALLOCATION & MODEL CREATION
# =============================================================================

# Automatic hardware acceleration detection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"\n[INFO] Allocating model on device: {device.upper()}")
print(f"[INFO] SFREQ: {SFREQ}Hz | Samples: {INPUT_WINDOW_SAMPLES}")

# Initialize the model architecture (16 channels for BIOT bipolar)
braindecode.create_model(
    n_chans=16,
    sfreq=SFREQ,
    input_window_samples=INPUT_WINDOW_SAMPLES,
    n_outputs=2,
    device=device
)


# =============================================================================
# 5. TRAINING LOOP
# =============================================================================

print("\n[INFO] Starting training phase...")

braindecode.train(
    emotion='excited',
    trial_duration_train=TRIAL_DURATION_SEC,
    epochs_duration_train=4,
    stride_train=4,
    trial_duration_test=TRIAL_DURATION_SEC,
    epochs_duration_test=4,
    stride_test=4
)

print("\n[SUCCESS] Pipeline execution completed.")