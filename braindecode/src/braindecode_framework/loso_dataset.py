import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
import os

class PrepareDataset:
    def __init__(self):
        self.loso_dataset = {}

    def extract_loso_dataset(self, n_split, plot):
        X_list, y_list, g_list, Channels_name, Info = [], [], [], [], []
        for user in self.dataset:
            for session in self.dataset[user]:
                ch_names = self.dataset[user][session]['epochs'].ch_names
                info = self.dataset[user][session]['epochs'].info
                X = self.dataset[user][session]['epochs'].get_data()
                y = self.dataset[user][session]['epochs'].events[:, 2]
                X_list.append(X)
                y_list.append(y)
                Channels_name.append(ch_names)
                Info.append(info)
                for idx in range(X.shape[0]):
                    g_list.append(user)



        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        subject_id = np.array(g_list)

        gkf = GroupKFold(n_splits=n_split)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subject_id), start=0):
            print(fold)
            path = f"./loso_file/fold_{fold}"
            os.makedirs(path, exist_ok=True)

            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            np.save(os.path.join(path, "X_train.npy"), X_train_fold)
            np.save(os.path.join(path, "y_train.npy"), y_train_fold)

            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]
            np.save(os.path.join(path, "X_test.npy"), X_test_fold)
            np.save(os.path.join(path, "y_test.npy"), y_test_fold)

            subject_train_fold = subject_id[train_idx]
            subject_test_fold = subject_id[test_idx]
            np.save(os.path.join(path, "subject_train.npy"), subject_train_fold)
            np.save(os.path.join(path, "subject_test.npy"), subject_test_fold)

            np.save(os.path.join(path, "ch_info.npy"), Channels_name)
            np.save(os.path.join(path, "info.npy"), Info)
