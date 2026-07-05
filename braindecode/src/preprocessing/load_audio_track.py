import librosa
from glob import glob

class LoadAudioTrack:
    def __init__(self):
        self.audio_track = {}


    def load_audio_track(self, path_in, tmin, tmax, path_out):
        data = glob(path_in + '/*')
        data.sort()

        for idx, track in enumerate(data):
            print(f'Extract data: {track}')
            audio, sr = librosa.load(track, sr=None, mono=True)

            self.audio_track[idx] = audio
