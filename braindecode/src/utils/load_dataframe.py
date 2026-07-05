import pickle as pkl


class LoadDataframe:
    def __init__(self):
        self.dataset = None

    def load_dataframe(self, path_in):
        self.dataset = pkl.load(open(path_in, "rb"))
