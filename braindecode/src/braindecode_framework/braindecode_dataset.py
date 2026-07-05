from braindecode.datasets import RawDataset, BaseConcatDataset

from braindecode.datasets import RawDataset, BaseConcatDataset


class BraindecodeDataset:
    def __init__(self):
        # Inizializziamo il dizionario delle emozioni
        self.braindecode_dataset = {
            'excited': None,
            'relaxation': None,
            'sad': None,
            'angry': None
        }

    def create_raw_concat_dataset(self):
        # Dizionario temporaneo per accumulare i RawDataset
        datasets = {
            'excited': [],
            'relaxation': [],
            'sad': [],
            'angry': []
        }

        # Assicurati che self.dataset esista (caricato precedentemente)
        for emotion in self.dataset:
            for user in self.dataset[emotion]:
                for session in self.dataset[emotion][user]:
                    raw = self.dataset[emotion][user][session]["raw"]

                    # Filtro lunghezza minima (es. 6400 campioni)
                    if raw.n_times > 6400:
                        print(f"Aggiunta sessione: {session} per utente: {user} (Emozione: {emotion})")
                        ds = RawDataset(
                            raw=raw,
                            description={
                                "subject": user,
                                "session": session
                            }
                        )
                        if emotion in datasets:
                            datasets[emotion].append(ds)

        # Creazione dei BaseConcatDataset
        for emotion in datasets:
            if len(datasets[emotion]) > 0:
                self.braindecode_dataset[emotion] = BaseConcatDataset(datasets[emotion])
            else:
                print(f"Attenzione: Nessun dato valido trovato per l'emozione {emotion}")
# class BraindecodeDataset:
#     def __init__(self):
#         self.braindecode_dataset = {
#             'excited': {},
#             'relaxation': {},
#             'sad': {},
#             'angry': {}
#         }
#
#     def create_raw_concat_dataset(self):
#         datasets = {
#             'excited': [],
#             'relaxation': [],
#             'sad': [],
#             'angry': []
#         }
#
#         for emotion in self.dataset:
#             for user in self.dataset[emotion]:
#                 for session in self.dataset[emotion][user]:
#                     raw = self.dataset[emotion][user][session]["raw"]
#                     print(f"user {user} session: {session}")
#
#                     if raw.n_times > 6400:
#                         ds = RawDataset(
#                             raw=raw,
#                             description={
#                                 "subject": user,
#                                 "session": session
#                             }
#                         )
#                         if emotion in datasets:
#                             datasets[emotion].append(ds)
#
#         for emotion in datasets:
#             self.braindecode_dataset[emotion] = BaseConcatDataset(datasets[emotion])