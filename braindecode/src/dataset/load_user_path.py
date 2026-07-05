from glob import glob


class LoadUserPath:
    def __init__(self):
        self.data_user = None

    def load_user_path(self, path_in):
        self.data_user = glob(path_in + '/*')
        self.data_user.sort()
        print()

