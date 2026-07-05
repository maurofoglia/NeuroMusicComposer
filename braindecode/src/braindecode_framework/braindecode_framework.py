from src.utils.load_dataframe import LoadDataframe
from src.braindecode_framework.loso_dataset import PrepareDataset
from src.braindecode_framework.braindecode_dataset import BraindecodeDataset
from src.braindecode_framework.braindecode_preprocessing import BraindecodePreprocessing
from src.braindecode_framework.plot_raw import PlotRaw
from src.utils.save_pkl_data import SavePklData
from src.braindecode_framework.model import Model
from src.braindecode_framework.braindecode_runner import BraindecodeModelRunner

# generic runner for Braindecode models, allows to specify the model name and parameters dynamically
class BrainDecodeFramework(LoadDataframe, PrepareDataset, BraindecodeDataset, BraindecodePreprocessing, PlotRaw,
                           Model, BraindecodeModelRunner, SavePklData):

    # dynamic initialization of the framework, allowing to specify the model name and parameters
    # CTNet is default, but can be changed to any other model supported by Braindecode (e.g., BIOT, REVE, Labram, BENDR)
    def __init__(self, model_name="CTNet", model_kwargs=None, tuning_strategy="full"):
        LoadDataframe.__init__(self)
        PrepareDataset.__init__(self)
        BraindecodeDataset.__init__(self)
        BraindecodePreprocessing.__init__(self)
        PlotRaw.__init__(self)
        Model.__init__(self)

        # Initialization generic runner for Braindecode models
        # Passiamo il nuovo parametro tuning_strategy al motore
        BraindecodeModelRunner.__init__(
            self,
            model_name=model_name,
            model_kwargs=model_kwargs,
            tuning_strategy=tuning_strategy
        )

        SavePklData.__init__(self)