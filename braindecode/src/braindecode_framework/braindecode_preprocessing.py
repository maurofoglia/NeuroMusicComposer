import numpy as np
import mne
import eegprep
from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess as bd_preprocess,  # Rinominato per evitare conflitti con il nome del nostro metodo
)


# ==============================================================================
# FUNZIONI HELPER PER IL PREPROCESSING
# ==============================================================================

def normalize_cbramod(raw):
    """
    CBraMod richiede una normalizzazione fissa basata su 100 µV.
    """
    data = raw.get_data()
    raw._data = data / 1e-4
    return raw


def normalize_raw_percentile(raw, percentile=95.0, eps=1e-8):
    """
    BIOT richiede una normalizzazione basata sul percentile.
    """
    data = raw.get_data()
    scale = np.percentile(np.abs(data), percentile, axis=1)
    scale = np.maximum(scale, eps)
    raw._data = data / scale[:, None]
    return raw


def normalize_labram(raw):
    """
    LaBraM richiede una normalizzazione fissa impostando l'unità a 0.1 mV (1e-4 V),
    mantenendo i valori principalmente nel range tra -1 e 1.
    """
    data = raw.get_data()
    raw._data = data / 1e-4
    return raw


def normalize_reve(raw):
    """
    REVE richiede una normalizzazione specifica:
    1. Normalizzazione Z-score calcolata sull'intera sessione.
    2. Clipping dei valori anomali che superano le 15 deviazioni standard.
    3. Conversione in float32.
    """
    data = raw.get_data()

    # Calcolo Z-score per ogni canale
    means = np.mean(data, axis=1, keepdims=True)
    stds = np.std(data, axis=1, keepdims=True)

    # Evitiamo divisioni per zero in caso di canali piatti
    stds[stds == 0] = 1e-8

    data_z = (data - means) / stds

    # Clipping a +/- 15 deviazioni standard
    data_clipped = np.clip(data_z, -15.0, 15.0)

    # REVE lavora in float32
    raw._data = data_clipped.astype(np.float32)
    return raw


def normalize_eegpt(raw):
    """
    EEGPT / BrainGPT richiede una normalizzazione z-score standard,
    senza correzioni aggiuntive.
    """
    data = raw.get_data()

    # Calcolo Z-score per ogni canale
    means = np.mean(data, axis=1, keepdims=True)
    stds = np.std(data, axis=1, keepdims=True)

    # Evitiamo divisioni per zero
    stds[stds == 0] = 1e-8

    raw._data = (data - means) / stds
    return raw


def normalize_luna(raw):
    """
    LUNA richiede una normalizzazione z-score indipendente
    per ogni canale all'interno del campione.
    """
    data = raw.get_data()

    # Calcolo Z-score per ogni canale
    means = np.mean(data, axis=1, keepdims=True)
    stds = np.std(data, axis=1, keepdims=True)

    # Evitiamo divisioni per zero in caso di canali piatti
    stds[stds == 0] = 1e-8

    raw._data = (data - means) / stds
    return raw


def normalize_signaljepa(raw):
    """
    SignalJEPA richiede esplicitamente la conversione in microvolt (µV).
    MNE lavora in Volt nativamente, quindi scaliamo di 10^6.
    """
    data = raw.get_data()
    raw._data = data * 1e6
    return raw


def normalize_bendr(raw):
    """
    BENDR richiede che ogni sequenza sia scalata linearmente in modo che
    i valori massimi e minimi siano rispettivamente 1 e -1, come da paper originale.
    """
    data = raw.get_data()

    # Calcoliamo minimo e massimo per ogni canale
    mins = np.min(data, axis=1, keepdims=True)
    maxs = np.max(data, axis=1, keepdims=True)

    # Calcoliamo il range evitando divisioni per zero sui canali piatti
    ranges = maxs - mins
    ranges[ranges == 0] = 1e-8

    # Trasformazione Min-Max nel range [-1, 1]
    # Formula: 2 * ((x - min) / (max - min)) - 1
    raw._data = 2 * ((data - mins) / ranges) - 1
    return raw


def apply_eegprep_clean(raw):
    """
    Ponte tra MNE e eegprep per pulizia artefatti (ASR + Channel Cleaning).
    Mantiene intatte le annotazioni per Braindecode.
    """
    eeg_dict = {
        'data': raw.get_data() * 1e6,  # Convertiamo in uV
        'srate': raw.info['sfreq'],
        'nbchan': len(raw.ch_names),
        'pnts': raw.n_times,
        'trials': 1,
        'xmin': 0.0,
        'xmax': (raw.n_times - 1) / raw.info['sfreq'],
        'chanlocs': [dict(labels=ch) for ch in raw.ch_names],
    }

    try:
        cleaned_dict, _, _, removed_mask = eegprep.clean_artifacts(
            eeg_dict,
            ChannelCriterion=0.95,
            LineNoiseCriterion=6.0,
            BurstCriterion=20.0,
            WindowCriterion='off',
            FlatlineCriterion=15.0,
            Highpass='off'
        )
    except Exception as e:
        print(f"Errore critico in clean_artifacts: {e}")
        return raw

    bad_chans = [raw.ch_names[i] for i, is_bad in enumerate(removed_mask) if is_bad]

    new_data = cleaned_dict['data'] / 1e6  # Torniamo in Volt
    new_info = raw.info.copy()
    new_raw = mne.io.RawArray(new_data, new_info)
    new_raw.set_annotations(raw.annotations)

    if bad_chans:
        new_raw.info['bads'] = bad_chans
        try:
            new_raw.set_montage('standard_1020')
            new_raw.interpolate_bads(reset_bads=True)
        except Exception as e:
            print(f"Errore interpolazione spaziale: {e}")
            pass

    return new_raw


# ==============================================================================
# CLASSE PRINCIPALE (ROUTER)
# ==============================================================================

class BraindecodePreprocessing:
    def __init__(self):
        self.braindecode_preprocessing = {
            "excited": {}, "relaxation": {}, "sad": {}, "angry": {}
        }
        # Parametri per la pipeline Standard
        self.factor_new = 1e-3
        self.init_block_size = 1000

    def preprocess(self, low_cut_hz=0.3, high_cut_hz=75.0):
        # 1. Recuperiamo in automatico il nome del modello (es. da BrainDecodeFramework)
        # Se non lo trova, usa "Standard" di default
        current_model = getattr(self, "model_name", "Standard")

        for emotion in self.braindecode_dataset:
            raw_dataset = self.braindecode_dataset[emotion]
            print(f"\n[{emotion}] Esecuzione preprocessing ottimizzato per: {current_model}...")

            for ds in raw_dataset.datasets:
                raw = ds.raw
                raw.pick(picks="eeg")

                # ==========================================
                # PIPELINE 1: CBraMod
                # ==========================================
                if current_model == "CBraMod":
                    raw.resample(200)
                    raw.notch_filter(freqs=50.0)
                    raw.filter(l_freq=low_cut_hz, h_freq=high_cut_hz, fir_design="firwin")
                    raw = normalize_cbramod(raw)

                # ==========================================
                # PIPELINE 2: BIOT (Gestisce anche il nome del Bridge)
                # ==========================================
                elif current_model in ["BIOT", "InterpolatedBIOT"]:
                    raw.resample(200)
                    raw.filter(l_freq=low_cut_hz, h_freq=high_cut_hz, fir_design="firwin")
                    raw = normalize_raw_percentile(raw, percentile=95.0)

                # labram
                elif current_model in ["Labram", "InterpolatedLaBraM", "InterpolatedLabram"]:
                    # 1. Filtro passa-banda (il paper usa 0.1 - 75 Hz fissi)
                    # Proteggiamo il low_cut nel caso vengano passati valori più alti
                    _low_cut = 0.1 if low_cut_hz > 0.1 else low_cut_hz
                    raw.filter(l_freq=_low_cut, h_freq=75.0, fir_design="firwin")

                    # 2. Filtro Notch a 50 Hz
                    raw.notch_filter(freqs=50.0)

                    # 3. Ricampionamento a 200 Hz
                    raw.resample(200)

                    # 4. Normalizzazione
                    raw = normalize_labram(raw)

                # ==========================================
                # PIPELINE: REVE
                # ==========================================
                elif current_model in ["REVE", "InterpolatedREVE"]:
                    # REVE ha frequenze fisse molto specifiche in pretraining (0.5 - 99.5)
                    _reve_low = 0.5
                    _reve_high = 99.5

                    raw.resample(200)
                    raw.filter(l_freq=_reve_low, h_freq=_reve_high, fir_design="firwin")
                    raw = normalize_reve(raw)

                # ==========================================
                # PIPELINE: EEGPT / BrainGPT
                # ==========================================
                elif current_model in ["EEGPT", "BrainGPT", "InterpolatedEEGPT"]:
                    # EEGPT/BrainGPT usa 256 Hz e filtro fisso 0.1 - 100 Hz
                    raw.resample(256)

                    _eegpt_low = 0.1 if low_cut_hz > 0.1 else low_cut_hz

                    raw.filter(l_freq=_eegpt_low, h_freq=100.0, fir_design="firwin")
                    raw = normalize_eegpt(raw)

                # ==========================================
                # PIPELINE: LUNA
                # ==========================================
                elif current_model in ["LUNA", "InterpolatedLUNA"]:
                    # LUNA usa 256 Hz, filtro passa-banda 0.1 - 75 Hz e Notch 50 Hz
                    raw.resample(256)

                    _luna_low = 0.1 if low_cut_hz > 0.1 else low_cut_hz

                    raw.filter(l_freq=_luna_low, h_freq=75.0, fir_design="firwin")
                    raw.notch_filter(freqs=50.0)

                    raw = normalize_luna(raw)

                # ==========================================
                # PIPELINE: SignalJEPA
                # ==========================================
                elif current_model in ["SignalJEPA", "InterpolatedSignalJEPA"]:
                    # Lasciamo la risoluzione temporale intatta qui per non rompere il windowing
                    # di Skorch. Il downsampling a 128Hz lo farà il nostro Bridge in GPU.
                    # Applichiamo i filtri richiesti dal paper: 0.5 - 40 Hz.
                    raw.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin")
                    raw = normalize_signaljepa(raw)

                # ==========================================
                # PIPELINE: BENDR
                # ==========================================
                elif current_model in ["BENDR", "InterpolatedBENDR"]:
                    # BENDR richiede resampling a 256 Hz e Min-Max scaling nel range [-1, 1]
                    raw.resample(256)
                    raw.filter(l_freq=low_cut_hz, h_freq=high_cut_hz, fir_design="firwin")
                    raw = normalize_bendr(raw)

                # ==========================================
                # PIPELINE 3: STANDARD (CTNet, EEGNet, ecc.)
                # ==========================================
                else:
                    raw.resample(200)
                    raw.filter(l_freq=low_cut_hz, h_freq=None, fir_design='firwin')
                    raw = apply_eegprep_clean(raw)
                    raw.filter(l_freq=None, h_freq=high_cut_hz, fir_design='firwin')

                # Salviamo il file RAW processato
                ds.raw = raw

            # Fase finale dedicata solo alla pipeline Standard (Standardizzazione Braindecode in-place)
            if current_model not in ["CBraMod", "BIOT", "InterpolatedBIOT", "REVE", "InterpolatedREVE", "EEGPT",
                                     "BrainGPT", "InterpolatedEEGPT", "LUNA", "InterpolatedLUNA", "SignalJEPA",
                                     "InterpolatedSignalJEPA", "BENDR", "InterpolatedBENDR"]:
                try:
                    preprocessors = [
                        Preprocessor(
                            exponential_moving_standardize,
                            factor_new=self.factor_new,
                            init_block_size=self.init_block_size,
                        )
                    ]
                    bd_preprocess(raw_dataset, preprocessors)
                except Exception as e:
                    print(f"Errore durante la standardizzazione braindecode per {emotion}: {e}")

            # Salviamo il dataset finale
            self.braindecode_preprocessing[emotion] = raw_dataset
            print(f"[{emotion}] Preprocessing per {current_model} completato con successo.")