from src.dataset.load_user_path import LoadUserPath
from src.dataset.load_emotive_raw_data import LoadEmotiveRawDataset
from src.utils.save_pkl_data import SavePklData

class Dataset(LoadUserPath, LoadEmotiveRawDataset, SavePklData):
    def __init__(self):
        LoadUserPath.__init__(self)
        LoadEmotiveRawDataset.__init__(self)
        SavePklData.__init__(self)
