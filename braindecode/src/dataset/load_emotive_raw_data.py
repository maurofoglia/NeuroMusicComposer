import glob
import re
import mne
import pyxdf
import numpy as np
from src.utils.metadata_utility import create_df
import os

def correct_session(data):
    if data[0]['time_series'].shape[1] == 19:
        marker = data[1]
        eeg = data[0]

        data[0] = marker
        data[1] = eeg

    return data


# def extract_session_number(session_name):
#     return int(re.search(r"\d+", session_name).group())

def extract_session_number(session_name):
    # Prende solo il nome della cartella finale (es. 'ses-S001') gestendo sia gli slash che i backslash
    folder_name = session_name.replace('\\', '/').split('/')[-1]
    return int(re.search(r"\d+", folder_name).group())

def extract_marker_onsets_from_xdf(marker_stream, eeg_stream):
    """
    Return marker onsets in seconds relative to the start of the EEG stream.

    Uses XDF/LSL timestamps:
        marker_stream["time_stamps"]
        eeg_stream["time_stamps"]

    Does NOT use any Unix/local timestamps that may be stored inside marker_stream["time_series"].
    """

    eeg_times = np.asarray(eeg_stream["time_stamps"], dtype=float)
    marker_times = np.asarray(marker_stream["time_stamps"], dtype=float)

    if len(eeg_times) == 0:
        raise ValueError("EEG stream contains no timestamps.")

    if len(marker_times) == 0:
        raise ValueError("Marker stream contains no timestamps.")

    eeg_start_time = eeg_times[0]

    marker_onsets = marker_times - eeg_start_time

    return marker_onsets


def build_mne_annotations_from_xdf_markers(marker_stream, eeg_stream, raw, emotion_marker):
    """
    Build MNE annotations aligning markers to the EEG stream.

    Typical case for the files this loader handles:
        - EEG recording starts at t = 0
        - fixation_cross_marker starts at 0
        - emotion_marker starts at the first XDF marker

    If the stream contains at least two markers:
        - marker 0 => fixation_cross_marker
        - marker 1 => emotion_marker

    If the stream contains only one marker:
        - fixation_cross_marker is from 0 until the marker
        - emotion_marker is from the marker until the end of the recording
    """

    sfreq = raw.info["sfreq"]
    recording_end = raw.n_times / sfreq

    eeg_times = np.asarray(eeg_stream["time_stamps"], dtype=float)
    marker_times = np.asarray(marker_stream["time_stamps"], dtype=float)

    eeg_start_time = eeg_times[0]
    marker_onsets = marker_times - eeg_start_time

    if len(marker_onsets) == 1:
        # Example case: a single marker indicates the start of the emotional phase.
        fixation_onset = 0.0
        emotion_onset = float(marker_onsets[0])

    else:
        # Case with two or more markers: use the first as fixation start and the second as emotion start.
        fixation_onset = float(marker_onsets[0])
        emotion_onset = float(marker_onsets[1])

    fixation_duration = emotion_onset - fixation_onset
    emotion_duration = recording_end - emotion_onset

    if fixation_duration < 0:
        raise ValueError(
            f"Negative fixation duration: fixation_onset={fixation_onset}, "
            f"emotion_onset={emotion_onset}"
        )

    if emotion_duration < 0:
        raise ValueError(
            f"Negative emotion duration: emotion_onset={emotion_onset}, "
            f"recording_end={recording_end}"
        )

    onset = [
        fixation_onset,
        emotion_onset
    ]

    duration = [
        fixation_duration,
        emotion_duration
    ]

    description = [
        "fixation_cross_marker",
        emotion_marker
    ]

    annotations = mne.Annotations(
        onset=onset,
        duration=duration,
        description=description
    )

    return annotations, marker_onsets


class LoadEmotiveRawDataset:
    def __init__(self):
        self.dataset = {
            'excited': {},
            'relaxation': {},
            'sad': {},
            'angry': {}
        }
        self.class1, self.class2, self.class3, self.class4 = create_df()

    def load_emotive_raw_data_xdf(self, plot):
        for user_file in self.data_user:
            user_session = glob.glob(user_file + '/*')
            user_session.sort()

            for session in user_session:
                print(session)

                path = session + '/eeg/'
                pat_session = glob.glob(path + '/*')

                data, info = pyxdf.load_xdf(pat_session[1])
                data = correct_session(data)

                marker_stream = data[0]
                eeg_stream = data[1]

                sfreq = float(eeg_stream['info']['nominal_srate'][0])

                eeg = eeg_stream['time_series']
                eeg = eeg[:, 3:17].T

                dataEEG = eeg * 1e-6

                # MUSE 2
                ch_names = ['AF7', 'AF8', 'TP9', 'TP10']


                # EMOTIV Epoc+ 14 channels (example):
                # ch_names = [
                #     'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                #     'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
                # ]

                ch_types = ['eeg'] * 14

                info = mne.create_info(
                    ch_names=ch_names,
                    sfreq=sfreq,
                    ch_types=ch_types
                )

                raw = mne.io.RawArray(dataEEG, info)
                raw.set_montage('standard_1020')

                session_number = extract_session_number(session) - 1

                emotion_marker = None

                if session_number in self.class1[0]:
                    emotion_marker = 'excited'

                if session_number in self.class2[0]:
                    emotion_marker = 'relaxation'

                if session_number in self.class3[0]:
                    emotion_marker = 'sad'

                if session_number in self.class4[0]:
                    emotion_marker = 'angry'


                eeg_times = np.asarray(eeg_stream["time_stamps"], dtype=float)
                marker_times = np.asarray(marker_stream["time_stamps"], dtype=float)

                if marker_times.size == 0:
                    print(f"Session skipped, no markers: {session}")
                    continue

                eeg_start_time = eeg_times[0]
                start_experiment = marker_times[0] - eeg_start_time
                trial_number_onset = start_experiment
                fixation_onset = start_experiment + 2.0
                fixation_onset = float(fixation_onset.item())
                stimulus_onset = start_experiment + 2.0 + 5.0
                stimulus_onset = float(stimulus_onset.item())


                onset = [
                    trial_number_onset,
                    fixation_onset,
                    stimulus_onset
                ]

                duration = [
                    0.01,
                    0.01,
                    0.01
                ]

                description = [
                    "trial_number",
                    "fixation_cross_marker",
                    emotion_marker
                ]

                annot_new = mne.Annotations(
                    onset=onset,
                    duration=duration,
                    description=description
                )

                raw.set_annotations(annot_new)

                if plot:
                    raw.plot(block=True)

                user_name = user_file.split('/')
                user_name = user_name[1].split('-')[1]
                session_number = session.split('/')[-1]

                if emotion_marker in self.dataset:
                    if user_name not in self.dataset[emotion_marker]:
                        self.dataset[emotion_marker][user_name] = {}

                    self.dataset[emotion_marker][user_name][session_number] = {
                        'raw': raw
                    }

    def load_emotive_raw_data_edf(self, plot=False):
        """
        New method to natively read .edf files with filtering for corrupted sessions.
        """

        # ==========================================
        # BLACKLIST OF CORRUPTED SESSIONS
        # Format: ("UserName", RealSessionNumber)
        # ==========================================
        corrupted_sessions = [
            ("ID019", 2),
            ("ID021", 23)
        ]

        for user_file in self.data_user:
            # Extract user_name robustly before iterating sessions
            # (Use replace('\\', '/') for safety on Windows)
            user_name = user_file.replace('\\', '/').split('/')[-1].split('-')[-1]

            user_session = glob.glob(user_file + '/*')
            user_session.sort()

            for session in user_session:
                # Extract the original session number (e.g. 2 or 23)
                real_session_number = extract_session_number(session)

                # 1. CHECK CORRUPTED SESSIONS
                if (user_name, real_session_number) in corrupted_sessions:
                    print(f"SKIPPED CORRUPTED SESSION: User {user_name} - Session {real_session_number}")
                    continue  # Skip directly to the next session!

                print(f"Processing EDF session: {session}")

                path = session + '/eeg/'
                pat_session = glob.glob(path + '/*.edf')

                if len(pat_session) == 0:
                    print(f"No .edf file found in {path}")
                    continue

                # 2. Automatic loading via MNE
                raw = mne.io.read_raw_edf(pat_session[0], preload=True)

                # 3. Set montage
                try:
                    raw.set_montage('standard_1020')
                except ValueError as e:
                    print(f"Unable to apply standard montage: {e}")

                # 4. Determine emotion
                # Subtract 1 because class indexing in create_df uses zero-based session indices
                session_number_idx = real_session_number - 1
                emotion_marker = None

                if session_number_idx in self.class1[0]:
                    emotion_marker = 'excited'
                elif session_number_idx in self.class2[0]:
                    emotion_marker = 'relaxation'
                elif session_number_idx in self.class3[0]:
                    emotion_marker = 'sad'
                elif session_number_idx in self.class4[0]:
                    emotion_marker = 'angry'

                if emotion_marker is None:
                    continue

                # 5. Create annotations (Markers)
                start_experiment = 0.0
                trial_number_onset = start_experiment
                fixation_onset = start_experiment + 2.0
                stimulus_onset = start_experiment + 2.0 + 5.0

                onset = [trial_number_onset, fixation_onset, stimulus_onset]
                duration = [0.01, 0.01, 0.01]
                description = ["trial_number", "fixation_cross_marker", emotion_marker]

                annot_new = mne.Annotations(
                    onset=onset,
                    duration=duration,
                    description=description
                )

                raw.set_annotations(annot_new)

                if plot:
                    raw.plot(block=True)

                # 6. Save into the dictionary
                session_number_str = session.split('/')[-1]

                if user_name not in self.dataset[emotion_marker]:
                    self.dataset[emotion_marker][user_name] = {}

                self.dataset[emotion_marker][user_name][session_number_str] = {
                    'raw': raw
                }


    def load_emotive_raw_data_fif(self, plot=False, use_zuna=False):
        """
        New method to natively read .fif files (e.g., those generated by ZUNA)
        with filtering for corrupted sessions.
        """

        # ==========================================
        # BLACKLIST OF CORRUPTED SESSIONS
        # Format: ("UserName", RealSessionNumber)
        # ==========================================
        corrupted_sessions = [
            ("ID019", 2),
            ("ID021", 23)
        ]

        for user_file in self.data_user:
            # Extract user_name robustly before iterating sessions
            user_name = user_file.replace('\\', '/').split('/')[-1].split('-')[-1]

            user_session = glob.glob(user_file + '/*')
            user_session.sort()

            for session in user_session:
                # Extract the original session number
                real_session_number = extract_session_number(session)

                # 1. CHECK CORRUPTED SESSIONS
                if (user_name, real_session_number) in corrupted_sessions:
                    print(f"SKIPPED CORRUPTED SESSION: User {user_name} - Session {real_session_number}")
                    continue

                print(f"Processing FIF session: {session}")

                path = session + '/eeg/'

                # Search for specific .fif files based on the use_zuna flag
                if use_zuna:
                    pat_session = glob.glob(path + '/*_zuna.fif')
                else:
                    # Search for standard .fif files and exclude _zuna.fif
                    pat_session = [f for f in glob.glob(path + '/*.fif') if '_zuna' not in f]

                if len(pat_session) == 0:
                    print(f"No .fif file found in {path}")
                    continue

                # 2. Automatic loading via MNE for FIF format
                # Select the first file found
                raw = mne.io.read_raw_fif(pat_session[0], preload=True, verbose='ERROR')

                # 3. Set montage
                try:
                    # raw.set_montage('standard_1020')
                    raw.set_montage('standard_1005')
                except ValueError as e:
                    print(f"Unable to apply standard montage: {e}")

                # 4. Determine emotion
                session_number_idx = real_session_number - 1
                emotion_marker = None

                if session_number_idx in self.class1[0]:
                    emotion_marker = 'excited'
                elif session_number_idx in self.class2[0]:
                    emotion_marker = 'relaxation'
                elif session_number_idx in self.class3[0]:
                    emotion_marker = 'sad'
                elif session_number_idx in self.class4[0]:
                    emotion_marker = 'angry'

                if emotion_marker is None:
                    continue

                # 5. Create annotations (Markers)
                # It is vital to reinsert them because ZUNA might have removed them during reconstruction
                start_experiment = 0.0
                trial_number_onset = start_experiment
                fixation_onset = start_experiment + 2.0
                stimulus_onset = start_experiment + 2.0 + 5.0

                onset = [trial_number_onset, fixation_onset, stimulus_onset]
                duration = [0.01, 0.01, 0.01]
                description = ["trial_number", "fixation_cross_marker", emotion_marker]

                annot_new = mne.Annotations(
                    onset=onset,
                    duration=duration,
                    description=description
                )

                raw.set_annotations(annot_new)

                if plot:
                    raw.plot(block=True)

                # 6. Save into the dictionary
                session_number_str = session.split('/')[-1]

                if user_name not in self.dataset[emotion_marker]:
                    self.dataset[emotion_marker][user_name] = {}

                self.dataset[emotion_marker][user_name][session_number_str] = {
                    'raw': raw
                }

    # def load_emotive_raw_data_fif(self, plot=False):
    #     """
    #     New method to natively read .fif files (e.g., those generated by ZUNA)
    #     with filtering for corrupted sessions.
    #     """
    #
    #     # ==========================================
    #     # BLACKLIST OF CORRUPTED SESSIONS
    #     # Format: ("UserName", RealSessionNumber)
    #     # ==========================================
    #     corrupted_sessions = [
    #         ("ID019", 2),
    #         ("ID021", 23)
    #     ]
    #
    #     for user_file in self.data_user:
    #         # Extract user_name robustly before iterating sessions
    #         user_name = user_file.replace('\\', '/').split('/')[-1].split('-')[-1]
    #
    #         user_session = glob.glob(user_file + '/*')
    #         user_session.sort()
    #
    #         for session in user_session:
    #             # Extract the original session number
    #             real_session_number = extract_session_number(session)
    #
    #             # 1. CHECK CORRUPTED SESSIONS
    #             if (user_name, real_session_number) in corrupted_sessions:
    #                 print(f"SKIPPED CORRUPTED SESSION: User {user_name} - Session {real_session_number}")
    #                 continue
    #
    #             print(f"Processing FIF session: {session}")
    #
    #             path = session + '/eeg/'
    #             # Cerchiamo i file .fif invece dei file .edf
    #             pat_session = glob.glob(path + '/*.fif')
    #
    #             if len(pat_session) == 0:
    #                 print(f"No .fif file found in {path}")
    #                 continue
    #
    #             # 2. Automatic loading via MNE for FIF format
    #             # Selezioniamo il primo file trovato (potresti voler gestire nomi specifici se hai sia fif grezzi che fif di zuna)
    #             raw = mne.io.read_raw_fif(pat_session[0], preload=True)
    #
    #             # 3. Set montage
    #             try:
    #                 raw.set_montage('standard_1020')
    #             except ValueError as e:
    #                 print(f"Unable to apply standard montage: {e}")
    #
    #             # 4. Determine emotion
    #             session_number_idx = real_session_number - 1
    #             emotion_marker = None
    #
    #             if session_number_idx in self.class1[0]:
    #                 emotion_marker = 'excited'
    #             elif session_number_idx in self.class2[0]:
    #                 emotion_marker = 'relaxation'
    #             elif session_number_idx in self.class3[0]:
    #                 emotion_marker = 'sad'
    #             elif session_number_idx in self.class4[0]:
    #                 emotion_marker = 'angry'
    #
    #             if emotion_marker is None:
    #                 continue
    #
    #             # 5. Create annotations (Markers)
    #             # È vitale reinserirli perché ZUNA potrebbe averli rimossi nella ricostruzione
    #             start_experiment = 0.0
    #             trial_number_onset = start_experiment
    #             fixation_onset = start_experiment + 2.0
    #             stimulus_onset = start_experiment + 2.0 + 5.0
    #
    #             onset = [trial_number_onset, fixation_onset, stimulus_onset]
    #             duration = [0.01, 0.01, 0.01]
    #             description = ["trial_number", "fixation_cross_marker", emotion_marker]
    #
    #             annot_new = mne.Annotations(
    #                 onset=onset,
    #                 duration=duration,
    #                 description=description
    #             )
    #
    #             raw.set_annotations(annot_new)
    #
    #             if plot:
    #                 raw.plot(block=True)
    #
    #             # 6. Save into the dictionary
    #             session_number_str = session.split('/')[-1]
    #
    #             if user_name not in self.dataset[emotion_marker]:
    #                 self.dataset[emotion_marker][user_name] = {}
    #
    #             self.dataset[emotion_marker][user_name][session_number_str] = {
    #                 'raw': raw
    #             }

    def expand_bids_with_fif(self, bids_root="BIDS"):
        """
        Esplora la cartella BIDS, trova i file .edf e salva una copia in formato .fif
        nella stessa cartella 'eeg'.
        """
        # Ora bids_root è una stringa corretta ("BIDS")
        search_pattern = os.path.join(bids_root, '**', 'eeg', '*.edf')
        edf_files = glob.glob(search_pattern, recursive=True)

        print(f"Trovati {len(edf_files)} file .edf da espandere.")

        for edf_path in edf_files:
            fif_path = edf_path.replace('.edf', '.fif')

            if os.path.exists(fif_path):
                print(f"File già presente, salto: {os.path.basename(fif_path)}")
                continue

            print(f"Elaborazione: {os.path.basename(edf_path)}")
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                raw.save(fif_path, overwrite=True, verbose=False)
                print(f"  -> Creato: {os.path.basename(fif_path)}")

            except Exception as e:
                print(f"  ❌ Errore con {edf_path}: {e}")