import mne

from src.utils.multifilter import multifilter


class MultiFilter:
    def __init__(self):
        self.multifilter_epochs = {}

    def multifilter(self, filters, range):
        for user in self.dataset_epochs:
            print(f'muultifiltering {user}')
            epochs = self.dataset_epochs[user]
            epochs_left = epochs['left']
            epochs_right = epochs['right']

            try:
                epochs_right = multifilter(epochs_right, frequencies=filters, freq_band=range)
                epochs_right = mne.make_fixed_length_epochs(epochs_right, duration=3, preload=True)
                epochs_left = multifilter(epochs_left, frequencies=filters, freq_band=range)
                epochs_left = mne.make_fixed_length_epochs(epochs_left, duration=3, preload=True)

                # Aggiungi i dati al dizionario
                if user not in self.multifilter_epochs:
                    self.multifilter_epochs[user] = {}
                    # Aggiungi il trial per l'utente specifico
                self.multifilter_epochs[user] = {
                    'left': epochs_left,
                    'right': epochs_right
                }
            except:
                ''