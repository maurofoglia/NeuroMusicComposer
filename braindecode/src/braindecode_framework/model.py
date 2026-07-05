from braindecode.models import CTNet


class Model:
    def __init__(self):
        self.device = None
        self.model = None
        self.n_outputs = None
        self.input_window_samples = None
        self.sfreq = None
        self.n_chans = None

    def create_model(self, n_chans, sfreq, input_window_samples, n_outputs, device):
        # model instance
        self.n_chans = n_chans
        self.sfreq = sfreq
        self.input_window_samples = input_window_samples
        self.n_outputs = n_outputs
        self.model = CTNet(
            n_chans=n_chans,
            sfreq=sfreq,
            n_times=input_window_samples,
            n_outputs=n_outputs,
        )
        self.device = device
        self.model = self.model.to(self.device)