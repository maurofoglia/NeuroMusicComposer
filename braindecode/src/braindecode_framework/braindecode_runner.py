import os
import random
import json
import numpy as np
import optuna
from optuna.integration import SkorchPruningCallback
import mne

import torch
import torch.nn as nn
import torch.nn.functional as F
import braindecode.models as braindecode_models

from huggingface_hub import hf_hub_download
import safetensors.torch
from safetensors.torch import load_file

from skorch.dataset import ValidSplit
from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping, LRScheduler

from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.util import set_random_seeds
from sklearn.metrics import confusion_matrix, classification_report

# Import modelli Braindecode
from braindecode.models import (
    BIOT, InterpolatedBIOT, CBraMod, Labram, InterpolatedLaBraM,
    REVE, EEGPT, LUNA, InterpolatedSignalJEPA, InterpolatedBENDR, SignalJEPA_PostLocal
)
from peft import LoraConfig, get_peft_model

# Scommenta questo blocco se utilizzi l'interpolazione spaziale per i Bridge
try:
    from braindecode.modules.interpolation import ChannelInterpolation
except ImportError:
    from braindecode.modules.interpolation import ChannelInterpolationLayer as ChannelInterpolation

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

set_random_seeds(
    seed=SEED,
    cuda=False,
    cudnn_benchmark=False
)

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Regex universale per colpire i layer lineari/attentivi della maggior parte dei Transformer
UNIVERSAL_LORA_TARGETS = r".*(qkv|proj|fc1|fc2|mlp\.dense|attention\.dense|wq|wk|wv|wo|linear|dense).*"


# ------------------------------------------------------------------
# BRIDGE 1: Per BIOT (Interpolazione Spaziale 4 -> 16 + Resampling)
# ------------------------------------------------------------------
class MusePRESTBridge(nn.Module):
    def __init__(self, n_outputs, tuning_strategy="full", **kwargs):
        super().__init__()
        self.target_sfreq = 200
        self.target_n_times = 800

        muse_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        info_muse = mne.create_info(ch_names=muse_channels, sfreq=self.target_sfreq, ch_types='eeg')
        info_muse.set_montage("standard_1020")

        prest_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'C4', 'T4', 'T5', 'P3', 'P4', 'T6', 'O1', 'O2']
        info_prest = mne.create_info(ch_names=prest_channels, sfreq=self.target_sfreq, ch_types='eeg')
        info_prest.set_montage("standard_1020")

        self.interpolator = ChannelInterpolation(info_muse["chs"], info_prest["chs"])

        self.biot = BIOT.from_pretrained(
            "braindecode/biot-pretrained-prest-16chs",
            n_outputs=n_outputs,
            n_times=self.target_n_times,
            sfreq=self.target_sfreq,
            strict=False,
            **kwargs
        )

        # ⚡ TUNING STRATEGY ⚡
        if tuning_strategy == "lora":
            print("[SISTEMA] Applicazione di LoRA al modello BIOT...")
            for param in self.interpolator.parameters():
                param.requires_grad = False
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=UNIVERSAL_LORA_TARGETS, lora_dropout=0.1, bias="none", modules_to_save=["final_layer"])
            self.biot = get_peft_model(self.biot, lora_config)
            self.biot.print_trainable_parameters()
        elif tuning_strategy == "linear_probing":
            print("[SISTEMA] Applicazione del Linear Probing a BIOT...")
            for param in self.parameters():
                param.requires_grad = False
            if hasattr(self.biot, 'final_layer'):
                for param in self.biot.final_layer.parameters():
                    param.requires_grad = True
        elif tuning_strategy == "full":
            print("[SISTEMA] BIOT caricato in modalità Full Fine-Tuning.")
        else:
            raise ValueError(f"Strategia di tuning '{tuning_strategy}' non riconosciuta.")

    def forward(self, x):
        x_200hz = F.interpolate(x, size=self.target_n_times, mode='linear', align_corners=False)
        x_16ch = self.interpolator(x_200hz)
        return self.biot(x_16ch)


# ------------------------------------------------------------------
# BRIDGE 2: Per CBraMod (Solo Resampling Temporale)
# ------------------------------------------------------------------
class MuseCBraModBridge(nn.Module):
    def __init__(self, n_outputs, n_chans=4, sfreq=200, n_times=800, tuning_strategy="full", **kwargs):
        super().__init__()
        self.target_sfreq = 200
        self.target_n_times = 800

        muse_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        info_muse = mne.create_info(ch_names=muse_channels, sfreq=self.target_sfreq, ch_types='eeg')
        info_muse.set_montage("standard_1020")

        self.cbramod = CBraMod.from_pretrained(
            "braindecode/cbramod-pretrained",
            n_outputs=n_outputs,
            n_chans=4,
            chs_info=info_muse["chs"],
            n_times=self.target_n_times,
            sfreq=self.target_sfreq,
            return_encoder_output=False,
            **kwargs
        )

        # ⚡ TUNING STRATEGY ⚡
        if tuning_strategy == "lora":
            print("[SISTEMA] Applicazione di LoRA al modello CBraMod...")
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=UNIVERSAL_LORA_TARGETS, lora_dropout=0.1, bias="none", modules_to_save=["final_layer"])
            self.cbramod = get_peft_model(self.cbramod, lora_config)
            self.cbramod.print_trainable_parameters()
        elif tuning_strategy == "linear_probing":
            print("[SISTEMA] Applicazione del Linear Probing a CBraMod...")
            for param in self.parameters():
                param.requires_grad = False
            if hasattr(self.cbramod, 'final_layer'):
                for param in self.cbramod.final_layer.parameters():
                    param.requires_grad = True
        elif tuning_strategy == "full":
            print("[SISTEMA] CBraMod caricato in modalità Full Fine-Tuning.")
        else:
            raise ValueError(f"Strategia '{tuning_strategy}' non riconosciuta.")

    def forward(self, x):
        return self.cbramod(x)


# ------------------------------------------------------------------
# BRIDGE 3: Per LaBraM PURO (Mappatura Spaziale Dinamica Nativa)
# ------------------------------------------------------------------
class MuseLabramBridge(nn.Module):
    def __init__(self, n_outputs, n_chans=4, sfreq=200, n_times=800, tuning_strategy="full", **kwargs):
        super().__init__()
        self.target_sfreq = 200
        self.target_n_times = 800
        self.muse_channels = ['TP9', 'AF7', 'AF8', 'TP10']

        info_muse = mne.create_info(ch_names=self.muse_channels, sfreq=self.target_sfreq, ch_types='eeg')
        info_muse.set_montage("standard_1020")

        self.labram = Labram.from_pretrained(
            "braindecode/labram-pretrained",
            n_outputs=n_outputs,
            n_chans=4,
            chs_info=info_muse["chs"],
            n_times=self.target_n_times,
            sfreq=self.target_sfreq,
            **kwargs
        )

        # ⚡ TUNING STRATEGY ⚡
        if tuning_strategy == "lora":
            print("[SISTEMA] Applicazione di LoRA al modello Labram...")
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=UNIVERSAL_LORA_TARGETS, lora_dropout=0.1, bias="none", modules_to_save=["final_layer"])
            self.labram = get_peft_model(self.labram, lora_config)
            self.labram.print_trainable_parameters()
        elif tuning_strategy == "linear_probing":
            print("[SISTEMA] Applicazione del Linear Probing a Labram...")
            for param in self.parameters():
                param.requires_grad = False
            if hasattr(self.labram, 'final_layer'):
                for param in self.labram.final_layer.parameters():
                    param.requires_grad = True
        elif tuning_strategy == "full":
            print("[SISTEMA] Labram caricato in modalità Full Fine-Tuning.")
        else:
            raise ValueError(f"Strategia '{tuning_strategy}' non riconosciuta.")

    def forward(self, x):
        return self.labram(x, ch_names=self.muse_channels)


# ------------------------------------------------------------------
# BRIDGE 5: Per EEGPT (Mappatura Spaziale Dinamica Nativa)
# ------------------------------------------------------------------
class MuseEEGPTBridge(nn.Module):
    def __init__(self, n_outputs, n_chans=4, sfreq=256, n_times=1024, tuning_strategy="full", **kwargs):
        super().__init__()
        self.target_sfreq = sfreq
        self.target_n_times = n_times

        muse_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        info_muse = mne.create_info(ch_names=muse_channels, sfreq=self.target_sfreq, ch_types='eeg')
        info_muse.set_montage("standard_1020")

        self.eegpt = EEGPT(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=info_muse["chs"],
            n_times=self.target_n_times,
            sfreq=self.target_sfreq,
            **kwargs
        )

        try:
            print("[SISTEMA] Download/Caricamento pesi EEGPT da HuggingFace...")
            model_file = hf_hub_download(repo_id="braindecode/eegpt-pretrained", filename="model.safetensors")
            state = load_file(model_file)

            clean_state = {}
            for k, v in state.items():
                if k.startswith("final_layer") or k.startswith("predictor") or k.startswith("reconstructor") or "chans_id" in k:
                    continue
                clean_state[k] = v
            self.eegpt.load_state_dict(clean_state, strict=False)
            print("[SISTEMA] Pesi EEGPT caricati con successo.")
        except Exception as e:
            print(f"[ATTENZIONE] Impossibile caricare i pesi di EEGPT: {e}")

        # ⚡ TUNING STRATEGY ⚡
        if tuning_strategy == "lora":
            print("[SISTEMA] Applicazione di LoRA al modello EEGPT...")
            # REGEX CORRETTA: Colpiamo solo i layer lineari dentro il Transformer,
            # ignorando esplicitamente le proiezioni dei canali iniziali.
            target_modules_regex = r".*attn\.(qkv|proj)|.*mlp\.(fc1|fc2).*"

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=target_modules_regex,
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["final_layer"]
            )

            # Wrappiamo il modello con PEFT
            self.eegpt = get_peft_model(self.eegpt, lora_config)
            self.eegpt.print_trainable_parameters()

        elif tuning_strategy == "linear_probing":
            print("[SISTEMA] Applicazione del Linear Probing a EEGPT...")
            for param in self.eegpt.parameters():
                param.requires_grad = False
            if hasattr(self.eegpt, 'final_layer'):
                for param in self.eegpt.final_layer.parameters():
                    param.requires_grad = True
        elif tuning_strategy == "full":
            print("[SISTEMA] EEGPT caricato in modalità Full Fine-Tuning.")
        else:
            raise ValueError(f"Strategia '{tuning_strategy}' non riconosciuta.")

    def forward(self, x):
        return self.eegpt(x)


# ------------------------------------------------------------------
# BRIDGE 6: Per LUNA (Topology-Agnostic / Nessuna Interpolazione)
# ------------------------------------------------------------------
class MuseLUNABridge(nn.Module):
    def __init__(self, n_outputs, n_chans=4, sfreq=256, n_times=1024, tuning_strategy="full", **kwargs):
        super().__init__()
        self.target_sfreq = sfreq
        self.target_n_times = n_times

        muse_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        info_muse = mne.create_info(ch_names=muse_channels, sfreq=self.target_sfreq, ch_types='eeg')
        info_muse.set_montage("standard_1020")

        self.luna = LUNA(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=self.target_n_times,
            sfreq=self.target_sfreq,
            chs_info=info_muse["chs"],
            **kwargs
        )

        try:
            print("[SISTEMA] Download/Caricamento pesi LUNA_base da HuggingFace...")
            model_path = hf_hub_download(repo_id="thorir/LUNA", filename="LUNA_base.safetensors")
            state_dict = load_file(model_path)

            mapping = self.luna.mapping.copy() if hasattr(self.luna, 'mapping') else {}
            mapping["cross_attn.temparature"] = "cross_attn.temperature"
            mapped_state_dict = {mapping.get(k, k): v for k, v in state_dict.items()}

            self.luna.load_state_dict(mapped_state_dict, strict=False)
            print("[SISTEMA] Pesi LUNA caricati con successo.")
        except Exception as e:
            print(f"[ATTENZIONE] Impossibile caricare i pesi di LUNA: {e}")

        # ⚡ TUNING STRATEGY ⚡
        if tuning_strategy == "lora":
            print("[SISTEMA] Applicazione di LoRA al modello LUNA...")
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=UNIVERSAL_LORA_TARGETS, lora_dropout=0.1, bias="none", modules_to_save=["final_layer"])
            self.luna = get_peft_model(self.luna, lora_config)
            self.luna.print_trainable_parameters()
        elif tuning_strategy == "linear_probing":
            print("[SISTEMA] Applicazione del Linear Probing a LUNA...")
            for param in self.luna.parameters():
                param.requires_grad = False
            if hasattr(self.luna, 'final_layer'):
                for param in self.luna.final_layer.parameters():
                    param.requires_grad = True
        elif tuning_strategy == "full":
            print("[SISTEMA] LUNA caricato in modalità Full Fine-Tuning.")
        else:
            raise ValueError(f"Strategia '{tuning_strategy}' non riconosciuta.")

    def forward(self, x):
        return self.luna(x)


# ------------------------------------------------------------------
# BRIDGE 7: Per SignalJEPA (Architettura Post-Local Manuale)
# ------------------------------------------------------------------
class MuseSignalJEPABridge(nn.Module):
    def __init__(self, n_outputs, tuning_strategy="full", **kwargs):
        super().__init__()
        self.target_sfreq = 128
        self.target_n_times = 512

        muse_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        info_muse = mne.create_info(ch_names=muse_channels, sfreq=self.target_sfreq, ch_types='eeg')
        info_muse.set_montage("standard_1020")

        self.jepa_backbone = InterpolatedSignalJEPA.from_pretrained(
            "braindecode/signal-jepa",
            chs_info=info_muse["chs"],
            n_times=self.target_n_times,
            sfreq=self.target_sfreq,
            strict=False,
            **kwargs
        )

        dummy_input = torch.zeros(1, 4, self.target_n_times)
        with torch.no_grad():
            dummy_output = self.jepa_backbone(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, n_outputs)
        )

        # ⚡ TUNING STRATEGY ⚡
        if tuning_strategy == "lora":
            print("[SISTEMA] Applicazione di LoRA al modello SignalJEPA...")
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=UNIVERSAL_LORA_TARGETS, lora_dropout=0.1, bias="none")
            self.jepa_backbone = get_peft_model(self.jepa_backbone, lora_config)
            self.jepa_backbone.print_trainable_parameters()
        elif tuning_strategy == "linear_probing":
            print("[SISTEMA] Applicazione del Linear Probing a SignalJEPA...")
            for param in self.jepa_backbone.parameters():
                param.requires_grad = False
            # La testa final_layer rimane addestrabile
        elif tuning_strategy == "full":
            print("[SISTEMA] SignalJEPA caricato in modalità Full Fine-Tuning.")
        else:
            raise ValueError(f"Strategia '{tuning_strategy}' non riconosciuta.")

    def forward(self, x):
        x_128hz = F.interpolate(x, size=self.target_n_times, mode='linear', align_corners=False)
        features = self.jepa_backbone(x_128hz)
        logits = self.final_layer(features)
        return logits


# ------------------------------------------------------------------
# BRIDGE 8: Per BENDR (BErt-inspired Neural Data Representations)
# ------------------------------------------------------------------
class MuseBENDRBridge(nn.Module):
    def __init__(self, n_outputs, n_chans=4, sfreq=256, n_times=1024, tuning_strategy="full", **kwargs):
        super().__init__()
        self.target_sfreq = sfreq
        self.target_n_times = n_times

        muse_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        info_muse = mne.create_info(ch_names=muse_channels, sfreq=self.target_sfreq, ch_types='eeg')
        info_muse.set_montage("standard_1020")

        self.bendr = InterpolatedBENDR(
            chs_info=info_muse["chs"],
            n_chans=19,
            n_outputs=n_outputs,
            n_times=self.target_n_times,
            sfreq=self.target_sfreq,
            **kwargs
        )

        try:
            print("[SISTEMA] Download/Caricamento pesi BENDR da HuggingFace...")
            try:
                model_file = hf_hub_download(repo_id="braindecode/braindecode-bendr", filename="model.safetensors")
                state = safetensors.torch.load_file(model_file)
            except Exception:
                model_file = hf_hub_download(repo_id="braindecode/braindecode-bendr", filename="pytorch_model.bin")
                state = torch.load(model_file, map_location="cpu")

            clean_state = {k: v for k, v in state.items() if not (k.startswith("final_layer") or k.startswith("clf"))}
            self.bendr.load_state_dict(clean_state, strict=False)
            print("[SISTEMA] Pesi BENDR caricati con successo.")
        except Exception as e:
            print(f"[ATTENZIONE] Impossibile caricare i pesi di BENDR: {e}")

        # ⚡ TUNING STRATEGY ⚡
        head_name = "final_layer" if hasattr(self.bendr, 'final_layer') else "clf"

        if tuning_strategy == "lora":
            print("[SISTEMA] Applicazione di LoRA al modello BENDR...")
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=UNIVERSAL_LORA_TARGETS, lora_dropout=0.1, bias="none", modules_to_save=[head_name])
            self.bendr = get_peft_model(self.bendr, lora_config)
            self.bendr.print_trainable_parameters()
        elif tuning_strategy == "linear_probing":
            print("[SISTEMA] Applicazione del Linear Probing a BENDR...")
            for param in self.bendr.parameters():
                param.requires_grad = False
            head = getattr(self.bendr, head_name, None)
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True
        elif tuning_strategy == "full":
            print("[SISTEMA] BENDR caricato in modalità Full Fine-Tuning.")
        else:
            raise ValueError(f"Strategia '{tuning_strategy}' non riconosciuta.")

    def forward(self, x):
        return self.bendr(x)


# ------------------------------------------------------------------
# BRIDGE 9: Per REVE con Tuning Strategy (LoRA, Linear Probing, Full)
# ------------------------------------------------------------------
class MuseREVEBridge(nn.Module):
    def __init__(self, n_outputs, n_chans=4, sfreq=256, n_times=1024, tuning_strategy="full", **kwargs):
        super().__init__()
        self.target_sfreq = sfreq
        self.target_n_times = n_times

        # IL BRIDGE E' L'UNICO RESPONSABILE DELLA GEOMETRIA
        muse_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        info_muse = mne.create_info(ch_names=muse_channels, sfreq=self.target_sfreq, ch_types='eeg')
        info_muse.set_montage("standard_1020")

        self.reve = REVE(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=info_muse["chs"],
            n_times=self.target_n_times,
            sfreq=self.target_sfreq,
            **kwargs
        )

        try:
            print("[SISTEMA] Download/Caricamento pesi REVE-Base da HuggingFace...")
            model_file = hf_hub_download(repo_id="brain-bzh/reve-base", filename="model.safetensors")
            state = load_file(model_file)

            clean_state = {}
            for k, v in state.items():
                if k.startswith("final_layer") or k.startswith("clf"):
                    continue
                if "position" in k and v.shape != getattr(self.reve, k, torch.empty(0)).shape:
                    continue
                clean_state[k] = v

            self.reve.load_state_dict(clean_state, strict=False)
            print("[SISTEMA] Pesi REVE caricati con successo.")
        except Exception as e:
            print(f"[ATTENZIONE] Impossibile caricare i pesi di REVE: {e}")

        # ⚡ TUNING STRATEGY ⚡
        if tuning_strategy == "lora":
            print("[SISTEMA] Applicazione di LoRA al modello REVE...")
            target_modules_regex = r".*(qkv|proj|fc1|fc2|mlp\.dense|attention\.dense).*"

            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules_regex, lora_dropout=0.1,
                                     bias="none", modules_to_save=["final_layer"])
            self.reve = get_peft_model(self.reve, lora_config)
            self.reve.print_trainable_parameters()

        elif tuning_strategy == "linear_probing":
            print("[SISTEMA] Applicazione del Linear Probing a REVE...")
            for param in self.reve.parameters():
                param.requires_grad = False
            for param in self.reve.final_layer.parameters():
                param.requires_grad = True

        elif tuning_strategy == "full":
            print("[SISTEMA] REVE caricato in modalità Full Fine-Tuning.")

        else:
            raise ValueError(f"Strategia di tuning '{tuning_strategy}' non riconosciuta.")

    def forward(self, x):
        return self.reve(x)

# ------------------------------------------------------------------
# CLASSE RUNNER PRINCIPALE UNIFICATA
# ------------------------------------------------------------------
class BraindecodeModelRunner:
    def __init__(self, model_name="CTNet", model_kwargs=None, tuning_strategy="full"):
        self.device = None
        self.n_outputs = None
        self.input_window_samples = None
        self.sfreq = None
        self.n_chans = None
        self.study = None

        self.model_name = model_name
        self.tuning_strategy = tuning_strategy

        self.foundation_models = [
            "InterpolatedBIOT", "CBraMod", "Labram", "InterpolatedLabram",
            "InterpolatedLaBraM", "EEGPT", "LUNA", "InterpolatedSignalJEPA",
            "InterpolatedBENDR", "REVE"
        ]

        if self.model_name == "InterpolatedBIOT":
            self.model_class = InterpolatedBIOT
        elif self.model_name == "CBraMod":
            self.model_class = CBraMod
        elif self.model_name in ["Labram", "InterpolatedLabram", "InterpolatedLaBraM"]:
            self.model_class = Labram if self.model_name == "Labram" else InterpolatedLaBraM
        elif self.model_name == "EEGPT":
            self.model_class = EEGPT
        elif self.model_name == "LUNA":
            self.model_class = LUNA
        elif self.model_name == "InterpolatedSignalJEPA":
            self.model_class = InterpolatedSignalJEPA
        elif self.model_name == "InterpolatedBENDR":
            self.model_class = InterpolatedBENDR
        elif self.model_name == "REVE":
            self.model_class = REVE
        else:
            self.model_class = getattr(braindecode_models, self.model_name)

        self.extra_model_kwargs = model_kwargs if model_kwargs else {}

    @staticmethod
    def extract_dataset_from_users(subject, dataset):
        concat_dataset = []
        for user in subject:
            subject_dataset = dataset[user]
            for user_session in subject_dataset.datasets:
                concat_dataset.append(user_session)
        concat_dataset = BaseConcatDataset(concat_dataset)
        return concat_dataset

    def extract_epochs(self, dataset, mapping, epoch_duration_s, trial_duration_s, stride_s):
        window_size_samples = int(epoch_duration_s * self.sfreq)
        trial_duration_samples = int(trial_duration_s * self.sfreq)
        window_stride_samples = int(stride_s * self.sfreq)

        epochs = create_windows_from_events(
            dataset,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=trial_duration_samples - 1,
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            drop_last_window=False,
            mapping=mapping,
        )

        return epochs

    def _get_model_kwargs(self, trial=None, best_params=None):
        kwargs = {
            "module__n_times": self.input_window_samples,
            "module__sfreq": self.sfreq,
            "module__n_chans": self.n_chans,
            "module__n_outputs": self.n_outputs,
        }

        for k, v in self.extra_model_kwargs.items():
            kwargs[f"module__{k}"] = v

        if self.model_name == "CTNet":
            if trial:
                kwargs["module__final_drop_prob"] = trial.suggest_float("final_drop_prob", 0.2, 0.8)
            elif best_params:
                kwargs["module__final_drop_prob"] = best_params["final_drop_prob"]
        elif self.model_name in ["EEGNetv4", "ShallowFBCSPNet", "Deep4Net"]:
            if trial:
                kwargs["module__drop_prob"] = trial.suggest_float("drop_prob", 0.2, 0.8)
            elif best_params:
                kwargs["module__drop_prob"] = best_params["drop_prob"]

        return kwargs

    def get_pretrained_module(self):
        clean_kwargs = {k.replace("module__", ""): v for k, v in self.extra_model_kwargs.items()}
        clean_kwargs.pop("chs_info", None)

        if self.model_name == "InterpolatedBIOT":
            return MusePRESTBridge(n_outputs=self.n_outputs, tuning_strategy=self.tuning_strategy, **clean_kwargs)
        elif self.model_name == "CBraMod":
            return MuseCBraModBridge(n_outputs=self.n_outputs, n_times=self.input_window_samples, sfreq=self.sfreq, tuning_strategy=self.tuning_strategy, **clean_kwargs)
        elif self.model_name == "Labram":
            return MuseLabramBridge(n_outputs=self.n_outputs, tuning_strategy=self.tuning_strategy, **clean_kwargs)
        elif self.model_name == "EEGPT":
            return MuseEEGPTBridge(n_outputs=self.n_outputs, n_chans=self.n_chans, sfreq=self.sfreq, n_times=self.input_window_samples, tuning_strategy=self.tuning_strategy, **clean_kwargs)
        elif self.model_name == "LUNA":
            return MuseLUNABridge(n_outputs=self.n_outputs, n_chans=self.n_chans, sfreq=self.sfreq, n_times=self.input_window_samples, tuning_strategy=self.tuning_strategy, **clean_kwargs)
        elif self.model_name == "InterpolatedSignalJEPA":
            return MuseSignalJEPABridge(n_outputs=self.n_outputs, tuning_strategy=self.tuning_strategy, **clean_kwargs)
        elif self.model_name == "InterpolatedBENDR":
            return MuseBENDRBridge(n_outputs=self.n_outputs, n_chans=self.n_chans, sfreq=self.sfreq, n_times=self.input_window_samples, tuning_strategy=self.tuning_strategy, **clean_kwargs)
        elif self.model_name == "REVE":
            return MuseREVEBridge(n_outputs=self.n_outputs, n_chans=self.n_chans, sfreq=self.sfreq, n_times=self.input_window_samples, tuning_strategy=self.tuning_strategy, **clean_kwargs)
        elif self.model_name in ["InterpolatedLabram", "InterpolatedLaBraM"]:
            chs_info = self.extra_model_kwargs.get("chs_info")
            if chs_info is None:
                muse_channels = ['TP9', 'AF7', 'AF8', 'TP10']
                info_muse = mne.create_info(ch_names=muse_channels, sfreq=self.sfreq if self.sfreq else 200, ch_types='eeg')
                info_muse.set_montage("standard_1020")
                chs_info = info_muse["chs"]

            model = InterpolatedLaBraM(
                n_outputs=self.n_outputs,
                n_chans=self.n_chans,
                chs_info=chs_info,
                n_times=self.input_window_samples,
                sfreq=self.sfreq,
                **clean_kwargs
            )

            try:
                model_file = hf_hub_download('braindecode/labram-pretrained', 'model.safetensors')
                state = safetensors.torch.load_file(model_file)
                state = {k: v for k, v in state.items() if k != 'temporal_embedding' and not k.startswith('head')}
                model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"Warning: could not fully load pretrained weights: {e}")

            if self.tuning_strategy == "lora":
                print("[SISTEMA] Applicazione di LoRA a InterpolatedLaBraM...")
                lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=UNIVERSAL_LORA_TARGETS, lora_dropout=0.1, bias="none", modules_to_save=["final_layer"])
                model = get_peft_model(model, lora_config)
            elif self.tuning_strategy == "linear_probing":
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.final_layer.parameters():
                    param.requires_grad = True

            return model
        else:
            raise ValueError(f"Nessun Bridge configurato per il modello {self.model_name}")

    def fit_best_model_and_evaluate(self, X_train, X_test, best_params):
        if self.model_name in self.foundation_models:
            module = self.get_pretrained_module()
            model_kwargs = {}
        else:
            module = self.model_class
            model_kwargs = self._get_model_kwargs(best_params=best_params)

        clf = EEGClassifier(
            module=module,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            train_split=ValidSplit(0.2, random_state=42),
            optimizer__lr=best_params["lr"],
            optimizer__weight_decay=best_params["weight_decay"],
            batch_size=best_params["batch_size"],
            max_epochs=300,
            callbacks=[
                "accuracy",
                ("lr_scheduler", LRScheduler(
                    policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    monitor="valid_loss",
                    factor=0.5,
                    patience=10,
                    min_lr=1e-5
                )),
                ("early_stopping", EarlyStopping(
                    patience=50,
                    monitor="valid_loss",
                    lower_is_better=True,
                    load_best=True
                )),
            ],
            device=self.device,
            classes=[0, 1],
            iterator_train__shuffle=True,
            iterator_train__num_workers=4,
            iterator_valid__num_workers=4,
            **model_kwargs
        )

        clf.fit(X_train, y=None)

        y_pred = clf.predict(X_test)
        y_true = X_test.get_metadata()["target"].to_numpy()

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true,
            y_pred,
            target_names=["fixation_cross_marker", "emotion"],
            output_dict=True
        )

        return clf, cm, report

    def objective(self, trial, X_train):
        if self.model_name in self.foundation_models:
            module = self.get_pretrained_module()
            model_kwargs = {}
        else:
            module = self.model_class
            model_kwargs = self._get_model_kwargs(trial=trial)

        clf = EEGClassifier(
            module=module,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            train_split=ValidSplit(0.2, random_state=42),
            optimizer__lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            optimizer__weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
            batch_size=trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
            max_epochs=300,
            callbacks=[
                "accuracy",
                ("lr_scheduler", LRScheduler(
                    policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    monitor="valid_loss",
                    factor=0.5,
                    patience=10,
                    min_lr=1e-5
                )),
                ("early_stopping", EarlyStopping(
                    patience=25,
                    monitor="valid_loss",
                    lower_is_better=True,
                    load_best=True
                )),
                ("pruning", SkorchPruningCallback(trial, monitor='valid_loss')),
            ],
            device=self.device,
            classes=[0, 1],
            iterator_train__shuffle=True,
            iterator_train__num_workers=4,
            iterator_valid__num_workers=4,
            **model_kwargs
        )

        clf.fit(X_train, y=None)
        valid_loss = clf.history[-1, "valid_loss"]

        return valid_loss

    def train(self, emotion, trial_duration_train, epochs_duration_train, stride_train,
              trial_duration_test, epochs_duration_test, stride_test):

        splits = self.dataset[emotion].split("subject")
        subject_keys = list(splits.keys())

        # Cartella di base per i risultati del modello corrente
        results_base_dir = os.path.join("results_loso", self.model_name)

        for i, test_subject in enumerate(subject_keys):

            # --- LOGICA DI RESUME AUTOMATICO ---
            subject_dir = os.path.join(results_base_dir, test_subject)
            expected_final_file = os.path.join(subject_dir, "classification_report.json")

            # Se il file finale esiste, il subject è già stato completato. Saltiamo al prossimo.
            if os.path.exists(expected_final_file):
                print(f"⏭️  [{self.model_name}] Soggetto {test_subject} già completato. Salto il training...")
                continue
            # -----------------------------------

            print(f"\n▶️  [{self.model_name}] Inizio training per il subject: {test_subject}")

            train_subject = [s for s in subject_keys if s != test_subject]

            test_set = splits[test_subject]
            train_set = self.extract_dataset_from_users(train_subject, splits)

            mapping = {
                "fixation_cross_marker": 0,
                emotion: 1,
            }

            X_train = self.extract_epochs(train_set, mapping,
                                          trial_duration_s=trial_duration_train,
                                          epoch_duration_s=epochs_duration_train, stride_s=stride_train)
            X_test = self.extract_epochs(test_set, mapping,
                                         trial_duration_s=trial_duration_test,
                                         epoch_duration_s=epochs_duration_test,
                                         stride_s=stride_test)

            study = optuna.create_study(
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
            )

            study.optimize(
                lambda trial: self.objective(trial, X_train),
                n_trials=30
            )

            print(f"[{self.model_name}] Test subject:", test_subject)
            print("Best params:", study.best_params)
            print("Best validation loss:", study.best_value)

            best_clf, cm, report = self.fit_best_model_and_evaluate(
                X_train=X_train,
                X_test=X_test,
                best_params=study.best_params
            )

            print("Confusion matrix:\n", cm)
            print("Classification report:\n", report)

            # Creazione cartella per il subject
            os.makedirs(subject_dir, exist_ok=True)

            # Salvataggio dei risultati usando os.path.join per pulizia e compatibilità
            with open(os.path.join(subject_dir, "classification_report.json"), "w") as f:
                json.dump(report, f, indent=4)

            with open(os.path.join(subject_dir, "confusion_matrix.json"), "w") as f:
                json.dump(cm.tolist(), f, indent=4)

            model_prefix = self.model_name.lower()
            best_clf.save_params(
                f_params=os.path.join(subject_dir, f"{model_prefix}_best_params.pt"),
                f_optimizer=os.path.join(subject_dir, f"{model_prefix}_best_optimizer.pt"),
                f_history=os.path.join(subject_dir, f"{model_prefix}_best_history.json")
            )

            print(f"Model saved to: {subject_dir}\n")