import numpy as np
import datetime as dt_
from datetime import datetime as dt

def correct_session(data):
    if data[0]['time_series'].shape[1] == 19:
        marker = data[1]
        eeg = data[0]

        data[0] = marker
        data[1] = eeg

    return data


def insert_marker(data):
    if data[0]['time_series'].shape[1] == 0:
        data[0]['time_series'] = np.array([1, 1.69694460e+09])
        data[0]['time_series'] = np.expand_dims(data[0]['time_series'], axis=0)
        # print('')

    if data[0]['time_series'].shape[0] > 1:
        data[0]['time_series'] = data[0]['time_series'][0,:]
        data[0]['time_series'] = np.expand_dims(data[0]['time_series'], axis=0)

    return data


def calculate_markers_(data):
    marker_timestamps = data[0]['time_series']
    marker_timestamps = marker_timestamps[:,1]

    dt_object = dt.fromtimestamp(int(marker_timestamps))
    marker_fixation = dt_object + dt_.timedelta(0,5)
    print('')
    return dt_object, marker_fixation