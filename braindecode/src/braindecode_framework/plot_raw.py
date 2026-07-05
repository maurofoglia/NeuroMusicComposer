


class PlotRaw:
    def __init__(self):
        pass

    def plot_raw(self, dataset, emotion):
        for i in range(len(dataset)):
            raw = dataset[emotion].datasets[i].raw.plot(block=True, scalings=dict(eeg=1))