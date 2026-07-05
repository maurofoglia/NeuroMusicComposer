import pickle as pkl
import mne
from dateutil import tz

from src.utils.create_mne_raw_data import create_raw, create_aligned_timestamps_marker, create_raw_marker
from eeglabio.utils import export_mne_raw
from datetime import datetime


def add_marker(raw):
    onsets = [0, 8]
    descriptions = ['fixation_cross', 'start_video']
    durations = 0.01

    # Crea annotazioni
    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw.set_annotations(annotations)
    raw.set_montage('standard_1020')

    return raw


class CreateRaw:
    def __init__(self):
        self.sfreq = 256
        self.dataset_raw = {}

    def create_raw(self, plot=False, path_dataset=None):
        dataset = pkl.load(open(path_dataset, 'rb'))
        for user in dataset:
            for session in dataset[user]:
                print(f'user {user} trial {session}')
                eeg = dataset[user][session]['eeg']

                if eeg.shape[0] > 1024:
                    raw = create_raw(EEG=eeg, sfreq=256)
                    raw = add_marker(raw=raw)

                    if plot:
                        raw.plot(block=True)

                    # Aggiungi i dati al dizionario
                    if user not in self.dataset_raw:
                        self.dataset_raw[user] = {}  # Crea una voce per l'utente se non esiste

                    # Aggiungi il trial per l'utente specifico
                    self.dataset_raw[user][session] = {'raw': raw.copy()}
                    print()
