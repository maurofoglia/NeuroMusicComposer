from src.utils.load_dataframe import LoadDataframe
from src.preprocessing.filter_raw import Filter
from src.preprocessing.extract_epochs import ExtractEpochs
from src.utils.save_pkl_data import SavePklData

class Preprocessing(LoadDataframe, Filter, ExtractEpochs, SavePklData):
    def __init__(self):
        LoadDataframe.__init__(self)
        Filter.__init__(self)
        ExtractEpochs.__init__(self)
        SavePklData.__init__(self)
