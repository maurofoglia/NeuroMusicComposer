import pandas as pd
import numpy as np


def create_df():
    video = pd.read_excel('Video_selezionati.xlsx')
    video = video.rename(columns=video.iloc[0]).drop(video.index[0]).reset_index(drop=True)
    # valence = video['AVG_Valence'].astype(float).to_numpy()
    # arousal = video['AVG_Arousal'].astype(float).to_numpy()
    quadranti = video['VAQ_Estimate'].to_numpy()

    classe_1 = np.where(quadranti == 1)
    classe_2 = np.where(quadranti == 2)
    classe_3 = np.where(quadranti == 3)
    classe_4 = np.where(quadranti == 4)

    # video = pd.read_excel('Self assessment_emotive 1.xls')
    # conditions = [
    #     (video['Valence'] >= 5) & (video['Arousal'] >= 5),
    #     (video['Valence'] >= 5) & (video['Arousal'] <= 5),
    #     (video['Valence'] <= 5) & (video['Arousal'] <= 5),
    #     (video['Valence'] <= 5) & (video['Arousal'] >= 5),
    # ]
    #
    # values = [1, 1, 2, 2]
    #
    # video['quadranti'] = np.select(conditions, values)

    return classe_1, classe_2, classe_3, classe_4


