import os
import re
from glob import glob
import pandas as pd
import numpy as np


class ReadMetrics:
    def __init__(self):
        self.cm_count = {}
        self.balance_accuracy = {}
        self.results_balance_accuracy = {}
        self.results_cm = {}

    def read_confusion_matrix_classification_report(self, path_in):
        data = glob(path_in + "/*.txt")
        data.sort()

        bal_acc_mean = []
        cm_positive = []
        for metrics in data:
            df = pd.read_csv(metrics, sep=';')
            col = df.columns[0]

            # trova la riga con "Confusion Matrix:"
            idx = df.index[df[col].astype(str).str.contains("Confusion Matrix", na=False)][0]
            cm_text = "\n".join(df.loc[idx + 1:idx + 5, col].astype(str).tolist())
            nums = list(map(int, re.findall(r"-?\d+", cm_text)))

            cm = np.array(nums[:4]).reshape(2, 2)

            if cm[0][0] > cm[0][1] and cm[1][0] < cm[1][1]:
                cm_positive.append(1)

            self.results_cm[metrics.split('/')[-1]] = [cm]

            text_all = "\n".join(df[col].astype(str).tolist())
            m = re.search(r"Balanced Accuracy:\s*([0-9]*\.?[0-9]+)", text_all)
            bal_acc = float(m.group(1)) if m else None
            bal_acc_mean.append(bal_acc)

        bal_acc_mean = sum(bal_acc_mean) / len(bal_acc_mean) if bal_acc_mean else None
        time_segments = metrics.split('/')[1]
        self.balance_accuracy[time_segments] = bal_acc_mean
        print(f"Balanced accuracy: {bal_acc_mean:.2f}")
        self.cm_count[time_segments] = [len(cm_positive)]
        print(f"Number of positive samples: {len(cm_positive)}")



    def accuracy_per_fold(self, path_in):
        data = glob(os.path.join(path_in, "*.txt"))
        data.sort()

        for metrics in data:
            df = pd.read_csv(metrics, sep=';')
            col = df.columns[0]

            text_all = "\n".join(df[col].astype(str).tolist())
            m = re.search(r"Balanced Accuracy:\s*([0-9]*\.?[0-9]+)", text_all)

            # >>> NUMERO (non stringa), già a 3 decimali
            bal_acc = round(float(m.group(1)), 3) if m else None

            fold_number = os.path.basename(metrics).split('_')[1]

            if fold_number not in self.results_balance_accuracy:
                self.results_balance_accuracy[fold_number] = {}

            # salva float -> Excel lo vede come numero
            self.results_balance_accuracy[fold_number] = {'balanced_accuracy': bal_acc}
