# import eegraph
import mne
import numpy as np
from mne.preprocessing import ICA, EOGRegression, Xdawn
from mne_icalabel import label_components
import pandas as pd
# from meegkit.detrend import reduce_ringing
# from pyod.models.knn import KNN
import datetime as dt_
from datetime import datetime as dt

def calculate_markers_(data):
    marker_timestamps = data[0]['time_series']
    marker_timestamps = marker_timestamps[:,1]

    dt_object = dt.fromtimestamp(int(marker_timestamps))
    marker_fixation = dt_object + dt_.timedelta(0,5)
    print('')
    return dt_object, marker_fixation

def calculate_markers(marker_1, marker_2):
    fixation_cross_marker = marker_1
    start_video_marker = marker_2

    return fixation_cross_marker, start_video_marker

def create_epochs_scratch(eeg_trasformed) -> object:
    # create raw
    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    ch_types = ['eeg'] * 14
    sfreq = 128.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    simulated_epochs = mne.EpochsArray(eeg_trasformed, info)
    simulated_epochs.set_montage('standard_1020')

    return simulated_epochs

def generate_raw(data):
    # create raw
    eeg = data['time_series']
    eeg = eeg[:,3:17].T
    dataEEG = eeg * 1e-6
    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    ch_types = ['eeg'] * 14
    sfreq = 128.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(dataEEG, info)
    raw.set_montage('standard_1020')
    # raw.plot()

    # set marker
    onset = min(raw.times) + 2
    onset = [onset, (onset + 20)]


    duration = 0.01  # Durations of the annotations in seconds. If a float, all the annotations are given the same duration.
    name = ['fixation_cross_marker', 'start_video_marker']
    annot_new = mne.Annotations(onset=onset,
                                duration=duration,
                                description=name)

    raw.set_annotations(annot_new)

    return raw
    # raw.plot()



# def remove_ring(epochs, contamination, threshold):
#     epochs_2 = epochs.copy()
#     epochs_2.filter(1, 40)
#     data = epochs_2.get_data()
#     data = np.squeeze(data).T
#     data = data * 1e6
#
#     # train kNN detector
#     clf = KNN(contamination=contamination)
#     clf.fit(data)
#
#     # get the prediction labels and outlier scores of the training data
#     y_train_scores = clf.decision_scores_  # raw outlier scores
#
#     # find index
#     # min_threshold = np.min(y_train_scores)
#     # max_threshold = np.max(y_train_scores)
#     # threshold = (min_threshold + max_threshold) / 2
#
#     idx = np.where(y_train_scores > threshold)[0]
#
#     if idx.shape[0] != 0:
#         data_2 = epochs.get_data()
#         data_2 = np.squeeze(data_2).T
#         data_2 = data_2 * 1e6
#         y = reduce_ringing(data_2, samples=idx, order=10, threshold=3)
#         y = y * 1e-6
#         y = y.T
#         y = np.expand_dims(y, axis=0)
#
#         ch_names = ['AF3', 'TP9', 'TP10', 'AF4']
#         ch_types = ['eeg', 'eeg', 'eeg', 'eeg']
#         sfreq = 256.0
#         info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
#         epochs_2 = mne.EpochsArray(y, info)
#         epochs_2.set_montage('standard_1020')
#         # epochs_2.plot()
#     else:
#         epochs_2 = epochs.copy()
#
#     return epochs_2


def preprocessing_eeg(epochs, prob_threshold):

    # ICA
    epochs_ica = epochs.copy()
    epochs_ica.filter(1, None)
    epochs.set_eeg_reference('average')

    ica = ICA(
        n_components=14,
        max_iter="auto",
        method="infomax",
        random_state=97,
        fit_params=dict(extended=True),
        verbose=1
    )
    ica.fit(epochs_ica)
    # ica.plot_sources(epochs)
    # ica.plot_components()
    # ica.apply(epochs, exclude=[0])
    #
    ic_labels = label_components(epochs_ica, ica, method="iclabel")
    labels = ic_labels["labels"]
    data_prob = pd.DataFrame(ic_labels)
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    data_prob = data_prob.loc[exclude_idx]
    rslt_df = data_prob.loc[data_prob['y_pred_proba'] > prob_threshold]
    exclude_idx = rslt_df.index
    ica.apply(epochs, exclude=exclude_idx)
    # epochs.plot()

    return epochs


def filter_raw(raw, l_freq, h_freq):
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    return raw


def create_epochs(raw, tmin, tmax):
    events, event_dict = mne.events_from_annotations(raw)

    event_mapping = {'fixation_cross_marker': 1, 'start_video_marker': 2}
    baseline = (None, 0)

    # Create epochs
    epochs = mne.Epochs(raw,
                        events, event_mapping,
                        tmin, tmax,
                        # baseline=(-0.1, 0),
                        baseline=None,
                        preload=True
                        )

    return epochs



def create_raw_to_connectivity(epochs):
    eeg_video = epochs.get_data().squeeze()

    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'P7', 'O1', 'O2', 'P8', 'FC6', 'F4', 'F8', 'AF4']
    ch_types = ['eeg'] * 12
    sfreq = 128.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_video, info)
    raw.set_montage('standard_1020')

    return raw


# def create_connectivity(raw, threshold, window):
#     path = 'test_data.edf'
#     mne.export.export_raw(path, raw, overwrite=True)
#
#     G = eegraph.Graph()
#     G.load_data(path=path)
#     graphs, connectivity_matrix = G.modelate(window_size=window, connectivity='pearson_correlation',
#                                              # bands=['delta','theta','alpha','beta','gamma'],
#                                              threshold=threshold)
#
#     # G.visualize_html(graphs[5], name='test_1')
#
#     return graphs, connectivity_matrix


def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]
