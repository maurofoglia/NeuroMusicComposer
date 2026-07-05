import asrpy
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components
import neurokit2 as nk
from src.utils.artifact_removal import ArtifactRemover
from src.utils.config import ProcessingConfig
import spkit as sp

class Filter:
    def __init__(self):
        self.chanset = None
        self.notch = None
        self.filt = None
        self.dataset_post_processing = {}

    def basic_preprocessing(self, plot, bad_channels_interpolation, asrpy_preprocessing, atar):

        for user in self.dataset:
            for sess in self.dataset[user]:
                print(f'preprocessing {user} {sess}')
                raw = self.dataset[user][sess]['raw'].copy()
                if raw.n_times > 5130:

                    if bad_channels_interpolation:
                        bads, info = nk.eeg_badchannels(raw, bad_threshold=0.5, distance_threshold=0.95, show=False)

                        raw.info['bads'] = bads

                        raw.interpolate_bads(reset_bads=True, mode='accurate')

                    if atar:
                        raw.filter(l_freq=0.5, h_freq=None, picks='eeg')
                        X = raw.get_data()

                        # EEG Channels
                        print(X.shape, X.min(), X.max())

                        # Scale in uV (order of 100s) and transpose (channle at axis =1)
                        X = X.T * 1e6

                        # ATAR
                        XR = sp.eeg.ATAR(X, wv='db3', winsize=128, thr_method='ipr', OptMode='elim', beta=0.1, k2=100,
                                         verbose=1, use_joblib=True)

                        # Scale and transpose back
                        XR = XR.T * 1e-6

                        raw = raw.copy()
                        raw._data = XR
                        # raw.plot(block=True)

                    if user not in self.dataset_post_processing:
                        self.dataset_post_processing[user] = {}

                    # Aggiungi il trial per l'utente specifico
                    self.dataset_post_processing[user][sess] = {'raw': raw}
























                    # if os.path.exists(path_out):
                    #     shutil.rmtree(path_out)  # elimina cartella e sottocartelle
                    #     print(f"Cartella '{path_out}' eliminata.")
                    # else:
                    #     print(f"La cartella '{path_out}' non esiste.")
                    #
                    # os.makedirs(path_out)
                    #
                    # # save raw
                    # base_out = Path(path_out)
                    # current_out = base_out / user / sess
                    # current_out.mkdir(parents=True, exist_ok=True)
                    # raw_clean.save(current_out / f"{user}_{sess}-raw.fif", overwrite=True)