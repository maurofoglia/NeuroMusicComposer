import numpy as np
from sklearn.decomposition import PCA
import mne


def smooth_fusing_epochs(sig1, sig2, smooth_para):
    sig1 = sig1.flatten()
    sig2 = sig2.flatten()
    m_point = (sig1[-1] + sig2[0]) / 2

    L1 = len(sig1)
    L1_half = int(np.floor(L1 / 2))
    L2 = len(sig2)
    L2_half = int(np.floor(L2 / 2))

    dif_1 = np.linspace(0, 1, L1_half) ** smooth_para
    dif_2 = np.linspace(1, 0, L2_half) ** smooth_para

    dif_m1 = sig1[-1] - m_point
    dif_m2 = sig2[0] - m_point

    sig1[L1_half + 0:] = sig1[L1_half + 0:] - dif_1.T * dif_m1
    sig2[:L2_half] = sig2[:L2_half] - dif_2 * dif_m2

    return sig1, sig2


def SPA_EEG(data, threshold, win_size, smooth_para, srate):
    s = win_size * srate
    segs = int(np.floor(data.shape[1] / s))

    data_new = data
    jj = 0
    print('SPA started.')
    for j in range(1, segs):
        jj += 1
        prog = int(jj * 100 / segs)
        if prog > 10:
            print(f'{int(j * 100 / segs)}%')
            jj = 0

        temp1 = data[:, (0 + (j - 1) * s):j * s]
        temp2 = data[:, (0 + j * s):(j + 1) * s]
        if j == segs - 1:
            temp2 = data[:, (0 + j * s):data.shape[1]]

        pca = PCA(n_components=None)
        b = pca.fit_transform(temp1.T) * -1
        a = pca.components_.T * -1
        c = pca.explained_variance_

        b[:, c > (threshold ** 2)] = 0
        temp1 = np.dot(b, a.T).T

        # pca = PCA(n_components=None)
        # b = pca.fit_transform(temp2.T) * -1
        # a = pca.components_.T * -1
        # c = pca.explained_variance_

        # b[:, c > (threshold ** 2)] = 0
        # temp2 = np.dot(b, a.T).T

        for c in range(data.shape[0]):
            sig_1, sig_2 = smooth_fusing_epochs(temp1[c, :], temp2[c, :], smooth_para)

            data_new[c, (0 + (j - 1) * s):j * s] = sig_1.T
            if j == segs - 1:
                data_new[c, (0 + j * s):data.shape[1]] = sig_2
            else:
                data_new[c, (0 + j * s):(j + 1) * s] = sig_2.T

    print('SPA completed.')
    return data_new


def SPA(raw):
    info = raw.info
    annotations = raw.annotations
    raw = raw.get_data().squeeze() * 1e6
    raw_preprocessed = SPA_EEG(data=raw, threshold=30, win_size=2, smooth_para=2,
                               srate=256) * 1e-6
    raw_preprocessed = mne.io.RawArray(raw_preprocessed, info)
    raw_preprocessed.set_annotations(annotations)
    return raw_preprocessed
