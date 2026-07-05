"""
Zuna Utility Module

This module contains the configuration constants and transformation functions
required to integrate the Zuna foundation model with the BIOT architecture.
It handles the conversion of Zuna physical channels into BIOT bipolar derivations.
"""

import mne
from src.dataset.dataset import Dataset

# =============================================================================
# 1. CONFIGURATION CONSTANTS (ZUNA -> BIOT)
# =============================================================================

ZUNA_PHYSICAL_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8'
]

BIOT_ANODES = [
    'Fp1', 'F7', 'T7', 'P7',
    'Fp2', 'F8', 'T8', 'P8',
    'Fp1', 'F3', 'C3', 'P3',
    'Fp2', 'F4', 'C4', 'P4'
]

BIOT_CATHODES = [
    'F7', 'T7', 'P7', 'O1',
    'F8', 'T8', 'P8', 'O2',
    'F3', 'C3', 'P3', 'O1',
    'F4', 'C4', 'P4', 'O2'
]

BIOT_BIPOLAR_NAMES = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'
]


# =============================================================================
# 2. TRANSFORMATION FUNCTIONS
# =============================================================================

def zuna_raw_to_biot_bipolar(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Converts a standard ZUNA raw EEG object into a BIOT-compatible bipolar raw object.

    Args:
        raw (mne.io.BaseRaw): The input raw EEG data containing ZUNA physical channels.

    Returns:
        mne.io.BaseRaw: A new MNE Raw object with 16 BIOT bipolar channels.

    Raises:
        ValueError: If any required ZUNA physical channels are missing in the input.
    """
    missing = [ch for ch in ZUNA_PHYSICAL_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing required ZUNA channels in the file: {missing}")

    # Preserve annotations across transformation
    annotations = raw.annotations.copy()

    # Pick and explicitly reorder physical channels to match ZUNA standards
    raw_phys = raw.copy().pick_channels(ZUNA_PHYSICAL_CHANNELS)
    raw_phys.reorder_channels(ZUNA_PHYSICAL_CHANNELS)

    # Compute bipolar derivations
    raw_bipolar = mne.set_bipolar_reference(
        raw_phys,
        anode=BIOT_ANODES,
        cathode=BIOT_CATHODES,
        ch_name=BIOT_BIPOLAR_NAMES,
        drop_refs=True,
        copy=True,
    )
    raw_bipolar.set_annotations(annotations)

    return raw_bipolar


def convert_dataset_to_biot_bipolar(dataset_obj: Dataset) -> None:
    """
    Iterates through the custom Dataset object and applies the Zuna-to-BIOT
    bipolar conversion in-place for all subjects, sessions, and emotions.

    Args:
        dataset_obj (Dataset): The initialized and loaded Dataset instance.
    """
    for emotion, users in dataset_obj.dataset.items():
        for user, sessions in users.items():
            for session, payload in sessions.items():
                raw = payload['raw']
                payload['raw'] = zuna_raw_to_biot_bipolar(raw)
                print(f"[PROCESS] Converted to BIOT Bipolar: emotion={emotion}, user={user}, session={session}")