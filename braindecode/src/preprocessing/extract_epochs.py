import mne
from mne.preprocessing import ICA, EOGRegression
from mne_icalabel import label_components
import os
import asrpy
from src.utils.artifact_removal import ArtifactRemover
from src.utils.config import ProcessingConfig

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

class ExtractEpochs:
    def __init__(self):
        self.dataset_epochs = {}

    def extract_epochs(self, tmin, tmax, LOW_FREQUENCY, HIGH_FREQUENCY, drop_channels, channels, eog, plot=False):
        for user in self.dataset_post_processing:
            for sess in self.dataset_post_processing[user]:
                print(f' user: {user} session: {sess}')
                raw = self.dataset_post_processing[user][sess]['raw'].copy()

                events, event_dict = mne.events_from_annotations(raw)

                event_mapping = {'fixation_cross_marker': 1, 'start_video_marker': 2}

                # Create epochs
                epochs = mne.Epochs(raw,
                                    events, event_mapping,
                                    tmin, tmax,
                                    baseline=None,
                                    preload=True
                                    )

                epochs = epochs.copy().filter(l_freq=None, h_freq=HIGH_FREQUENCY, verbose=True)

                if eog:
                    epochs = epochs.set_eeg_reference("average", projection=False)
                    epochs.set_channel_types({"AF3": "eog", "AF4": "eog"})
                    eog = EOGRegression()
                    eog.fit(epochs)
                    epochs = eog.apply(epochs)
                    epochs.set_eeg_reference("average", projection=True)

                if drop_channels:
                    epochs.drop_channels(channels)

                if epochs['fixation_cross_marker'] and epochs['start_video_marker']:

                    if plot:
                        epochs.plot(block=True, n_epochs=4)

                    # Aggiungi i dati al dizionario
                    if user not in self.dataset_epochs:
                        self.dataset_epochs[user] = {}

                    self.dataset_epochs[user][sess] = {'epochs': epochs}

