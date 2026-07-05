import pickle as pkl
import os


class SavePklData:

    def save_pkl_data(self, path_out, data):
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pkl.dump(data, open(path_out, 'wb'))
