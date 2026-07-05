import json
import os
import numpy as np
import optuna
import torch
from braindecode import EEGClassifier
from sklearn.metrics import balanced_accuracy_score
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.helper import predefined_split


class BraindecodeOptuna:
    # Rename the internal variable from 'braindecode_ctnet' to a generic 'runner'
    def __init__(self, runner):
        self.runner = runner
        self.random_state = 42
        self.study = None

    def create_study(self, study_name=None, storage=None):
        self.study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True if study_name is not None and storage is not None else False,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        )
        return self.study

    def suggest_params(self, trial):
        params = {
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
            "n_epochs": trial.suggest_categorical("n_epochs", [50, 100, 150, 200, 300]),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        }

        # Dynamically suggest the correct dropout parameter depending on the model
        model_name = self.runner.model_name

        if model_name == "CTNet":
            # --- 1. OTTIMIZZAZIONE DEI DROPOUT ---
            params["final_drop_prob"] = trial.suggest_float("final_drop_prob", 0.1, 0.8)
            params["cnn_drop_prob"] = trial.suggest_float("cnn_drop_prob", 0.1, 0.6)
            params["att_positional_drop_prob"] = trial.suggest_float("att_positional_drop_prob", 0.0, 0.5)

            # --- 2. OTTIMIZZAZIONE DEL TRANSFORMER ---
            params["num_heads"] = trial.suggest_categorical("num_heads", [2, 4, 8])
            params["num_layers"] = trial.suggest_int("num_layers", 2, 8)
            params["embed_dim"] = trial.suggest_categorical("embed_dim", [20, 40, 80])

            # --- 3. OTTIMIZZAZIONE DELLA CNN ---
            params["n_filters_time"] = trial.suggest_int("n_filters_time", 10, 40)
            params["kernel_size"] = trial.suggest_categorical("kernel_size", [32, 64, 128])
            params["depth_multiplier"] = trial.suggest_int("depth_multiplier", 1, 4)
            params["pool_size_1"] = trial.suggest_categorical("pool_size_1", [4, 8])
            params["pool_size_2"] = trial.suggest_categorical("pool_size_2", [4, 8])
        elif model_name in ["EEGNetv4", "ShallowFBCSPNet", "Deep4Net"]:
            params["drop_prob"] = trial.suggest_float("drop_prob", 0.2, 0.8)


        elif model_name == "REVE":
            # --- 1. TRANSFORMER STRUCTURE OPTIMIZATION ---
            # Reduce the search range compared to huge models to avoid excessive memory/compute usage
            params["embed_dim"] = trial.suggest_categorical("embed_dim", [128, 256, 512])
            params["depth"] = trial.suggest_int("depth", 4, 24)
            params["heads"] = trial.suggest_categorical("heads", [4, 8, 16])
            params["mlp_dim_ratio"] = trial.suggest_float("mlp_dim_ratio", 2.0, 4.0)

            # --- 2. ACTIVATION AND POOLING OPTIMIZATION ---
            params["use_geglu"] = trial.suggest_categorical("use_geglu", [True, False])
            params["attention_pooling"] = trial.suggest_categorical("attention_pooling", [True, False])

            # --- 3. PATCHING HYPERPARAMETERS ---
            # Use only multiples of the window size to avoid "RuntimeError shape invalid"
            params["patch_size"] = trial.suggest_categorical("patch_size", [64, 128, 256])
            params["patch_overlap"] = trial.suggest_categorical("patch_overlap", [0, 10, 20, 32])

        elif model_name == "BIOT":
            params["num_layers"] = trial.suggest_int("num_layers", 2, 8)

            embed_and_heads = trial.suggest_categorical("embed_and_heads", [
                (128, 4), (256, 8), (512, 8)
            ])
            params["embed_dim"] = embed_and_heads[0]
            params["num_heads"] = embed_and_heads[1]

            params["hop_length"] = trial.suggest_categorical("hop_length", [64, 128, 256])

            params["drop_prob"] = trial.suggest_float("drop_prob", 0.1, 0.6)
            params["att_drop_prob"] = trial.suggest_float("att_drop_prob", 0.0, 0.4)
            params["att_layer_drop_prob"] = trial.suggest_float("att_layer_drop_prob", 0.0, 0.4)


        elif model_name == "Labram":
            # --- 1. OTTIMIZZAZIONE STRUTTURA TRANSFORMER ---
            # Permettiamo a Optuna di rimpicciolire il modello per evitare overfitting
            params["num_layers"] = trial.suggest_int("num_layers", 4, 12)
            # Attenzione: in Labram, embed_dim DEVE essere divisibile per num_heads!
            # Creiamo combinazioni "sicure"
            embed_and_heads = trial.suggest_categorical("embed_and_heads", [
                (100, 5),  # 100 dim, 5 heads (100/5 = 20)
                (200, 10),  # 200 dim, 10 heads (default)
                (300, 10)  # 300 dim, 10 heads
            ])
            params["embed_dim"] = embed_and_heads[0]
            params["num_heads"] = embed_and_heads[1]
            params["mlp_ratio"] = trial.suggest_float("mlp_ratio", 2.0, 4.0)

            # --- 2. OTTIMIZZAZIONE DEL PATCHING E POOLING ---
            params["learned_patcher"] = trial.suggest_categorical("learned_patcher", [True, False])
            params["use_mean_pooling"] = trial.suggest_categorical("use_mean_pooling", [True, False])
            # Divisori perfetti di 512
            params["patch_size"] = trial.suggest_categorical("patch_size", [64, 128, 256])

            # --- 3. OTTIMIZZAZIONE DROPOUT ---
            params["drop_prob"] = trial.suggest_float("drop_prob", 0.0, 0.4)
            params["attn_drop_prob"] = trial.suggest_float("attn_drop_prob", 0.0, 0.4)
            params["drop_path_prob"] = trial.suggest_float("drop_path_prob", 0.0, 0.2)


        elif model_name == "BENDR":
            # --- 1. OTTIMIZZAZIONE ARCHITETTURA CENTRALE ---
            # Lasciamo decidere a Optuna se usare il Transformer o fermarsi all'Encoder CNN!
            params["encoder_only"] = trial.suggest_categorical("encoder_only", [True, False])

            # --- 2. DIMENSIONI DEI VETTORI ---
            params["encoder_h"] = trial.suggest_categorical("encoder_h", [256, 512, 1024])

            # --- 3. OTTIMIZZAZIONE DEL TRANSFORMER (Se usato) ---
            params["transformer_layers"] = trial.suggest_int("transformer_layers", 2, 8)
            params["transformer_heads"] = trial.suggest_categorical("transformer_heads", [4, 8])

            # --- 4. DROPOUT E REGOLARIZZAZIONE ---
            params["drop_prob"] = trial.suggest_float("drop_prob", 0.0, 0.3)
            # LayerDrop: utile solo se hai molti transformer_layers
            params["layer_drop"] = trial.suggest_float("layer_drop", 0.0, 0.1)


        elif model_name == "InterpolatedBIOT":
            # Nessun parametro strutturale! La rete è già pre-addestrata e fissa.
            # Optuna ottimizzerà solo LR, Weight Decay, Batch Size e N_epochs
            # (che sono già definiti all'inizio della funzione suggest_params).
            pass

        return params

    def prepare_model(self, n_chans, sfreq, input_window_samples, n_outputs):
        self.runner.n_chans = n_chans
        self.runner.sfreq = sfreq
        self.runner.input_window_samples = input_window_samples
        self.runner.n_outputs = n_outputs

        if self.runner.model_name == "InterpolatedBIOT":
            self.runner.model = self.runner.get_pretrained_module().to(self.runner.device)
        else:
            self.runner.model = self.runner.model_class(
                n_chans=n_chans,
                sfreq=sfreq,
                n_times=input_window_samples,
                n_outputs=n_outputs,
                **self.runner.extra_model_kwargs
            ).to(self.runner.device)

    def get_dataset(self, emotion):
        return self.runner.dataset[emotion]

    def get_mapping(self, emotion):
        mapping = {
            "fixation_cross_marker": 0,
            emotion: 1,
        }
        return mapping

    def split_subjects(self, subject_keys, test_subject):
        train_subject = [s for s in subject_keys if s != test_subject]
        valid_subject = train_subject[0:1]
        train_subjects = train_subject[1:]
        return train_subjects, valid_subject

    def create_train_dataset(self, splits, train_subjects):
        train_set = self.runner.extract_dataset_from_users(train_subjects, splits)
        return train_set

    def create_valid_dataset(self, splits, valid_subject):
        valid_set = self.runner.extract_dataset_from_users(valid_subject, splits)
        return valid_set

    def create_test_dataset(self, splits, test_subject):
        test_set = splits[test_subject]
        return test_set

    def create_train_epochs(self, train_set, mapping):
        x_train = self.runner.extract_epochs(
            train_set,
            mapping,
            trial_duration_s=6,
            epoch_duration_s=5,
            stride_s=0.5,
        )
        return x_train

    def create_valid_epochs(self, valid_set, mapping):
        x_valid = self.runner.extract_epochs(
            valid_set,
            mapping,
            trial_duration_s=6,
            epoch_duration_s=5,
            stride_s=0.5,
        )
        return x_valid

    def create_test_epochs(self, test_set, mapping):
        x_test = self.runner.extract_epochs(
            test_set,
            mapping,
            trial_duration_s=5,
            epoch_duration_s=5,
            stride_s=5,
        )
        return x_test

    def prepare_fold_data(self, splits, subject_keys, test_subject, emotion):
        train_subjects, valid_subject = self.split_subjects(subject_keys, test_subject)

        train_set = self.create_train_dataset(splits, train_subjects)
        valid_set = self.create_valid_dataset(splits, valid_subject)
        test_set = self.create_test_dataset(splits, test_subject)

        mapping = self.get_mapping(emotion)

        x_train = self.create_train_epochs(train_set, mapping)
        x_valid = self.create_valid_epochs(valid_set, mapping)
        x_test = self.create_test_epochs(test_set, mapping)

        return x_train, x_valid, x_test

    def create_callbacks(self):
        callbacks = [
            "accuracy",
            (
                "lr_scheduler",
                LRScheduler(
                    policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    monitor="valid_loss",
                    factor=0.5,
                    patience=10,
                    min_lr=1e-5,
                ),
            ),
            (
                "early_stopping",
                EarlyStopping(
                    patience=200,
                    monitor="valid_loss",
                    lower_is_better=True,
                    load_best=True,
                ),
            ),
        ]
        return callbacks

    def create_classifier(self, params, x_valid):
        if self.runner.model_name == "InterpolatedBIOT":
            module = self.runner.get_pretrained_module()
            model_kwargs = {}  # Nessun parametro extra
        else:
            module = self.runner.model_class
            # Prepare standard kwargs required by all models
            model_kwargs = {
                "module__n_times": self.runner.input_window_samples,
                "module__sfreq": self.runner.sfreq,
                "module__n_chans": self.runner.n_chans,
                "module__n_outputs": self.runner.n_outputs,
            }

            # Handle dropout parameter dynamically depending on the used model
            if self.runner.model_name == "CTNet":
                model_kwargs["module__final_drop_prob"] = params["final_drop_prob"]
                model_kwargs["module__cnn_drop_prob"] = params["cnn_drop_prob"]
                model_kwargs["module__att_positional_drop_prob"] = params["att_positional_drop_prob"]
                model_kwargs["module__num_heads"] = params["num_heads"]
                model_kwargs["module__num_layers"] = params["num_layers"]
                model_kwargs["module__embed_dim"] = params["embed_dim"]
                model_kwargs["module__n_filters_time"] = params["n_filters_time"]
                model_kwargs["module__kernel_size"] = params["kernel_size"]
                model_kwargs["module__depth_multiplier"] = params["depth_multiplier"]
                model_kwargs["module__pool_size_1"] = params["pool_size_1"]
                model_kwargs["module__pool_size_2"] = params["pool_size_2"]

            elif self.runner.model_name in ["EEGNetv4", "ShallowFBCSPNet", "Deep4Net"]:
                model_kwargs["module__drop_prob"] = params["drop_prob"]

            elif self.runner.model_name == "REVE":
                model_kwargs["module__embed_dim"] = params["embed_dim"]
                model_kwargs["module__depth"] = params["depth"]
                model_kwargs["module__heads"] = params["heads"]
                model_kwargs["module__mlp_dim_ratio"] = params["mlp_dim_ratio"]
                model_kwargs["module__use_geglu"] = params["use_geglu"]
                model_kwargs["module__attention_pooling"] = params["attention_pooling"]
                model_kwargs["module__patch_size"] = params["patch_size"]
                model_kwargs["module__patch_overlap"] = params["patch_overlap"]

            elif self.runner.model_name == "Labram":
                model_kwargs["module__num_layers"] = params["num_layers"]
                model_kwargs["module__embed_dim"] = params["embed_dim"]
                model_kwargs["module__num_heads"] = params["num_heads"]
                model_kwargs["module__mlp_ratio"] = params["mlp_ratio"]

                model_kwargs["module__learned_patcher"] = params["learned_patcher"]
                model_kwargs["module__use_mean_pooling"] = params["use_mean_pooling"]
                model_kwargs["module__patch_size"] = params["patch_size"]

                model_kwargs["module__drop_prob"] = params["drop_prob"]
                model_kwargs["module__attn_drop_prob"] = params["attn_drop_prob"]
                model_kwargs["module__drop_path_prob"] = params["drop_path_prob"]

            elif self.runner.model_name == "BENDR":
                model_kwargs["module__encoder_only"] = params["encoder_only"]
                model_kwargs["module__encoder_h"] = params["encoder_h"]
                model_kwargs["module__transformer_layers"] = params["transformer_layers"]
                model_kwargs["module__transformer_heads"] = params["transformer_heads"]
                model_kwargs["module__drop_prob"] = params["drop_prob"]
                model_kwargs["module__layer_drop"] = params["layer_drop"]

            elif self.runner.model_name == "BIOT":
                model_kwargs["module__num_layers"] = params["num_layers"]
                model_kwargs["module__embed_dim"] = params["embed_dim"]
                model_kwargs["module__num_heads"] = params["num_heads"]
                model_kwargs["module__hop_length"] = params["hop_length"]
                model_kwargs["module__drop_prob"] = params["drop_prob"]
                model_kwargs["module__att_drop_prob"] = params["att_drop_prob"]
                model_kwargs["module__att_layer_drop_prob"] = params["att_layer_drop_prob"]

            # Add any extra user-provided model kwargs from runner.extra_model_kwargs
            for k, v in self.runner.extra_model_kwargs.items():
                model_kwargs[f"module__{k}"] = v

        clf = EEGClassifier(
            module,
            #self.runner.model_class,  # <-- Passiamo la classe dinamicamente!
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            train_split=predefined_split(x_valid),
            optimizer__lr=params["lr"],
            optimizer__weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            max_epochs=params["n_epochs"],
            callbacks=self.create_callbacks(),
            device=self.runner.device,
            classes=[0, 1],
            iterator_train__shuffle=True,
            **model_kwargs  # Unpack dei parametri del modello
        )
        return clf

    def fit_classifier(self, clf, x_train):
        clf.fit(x_train, y=None)
        return clf

    def move_classifier_to_cpu(self, clf):
        cpu = torch.device("cpu")
        clf.set_params(device=cpu)
        clf.module_.to(cpu)
        clf.module_.eval()
        return clf

    def get_y_true(self, x_test):
        y_true = np.array([x_test[i][1] for i in range(len(x_test))])
        return y_true

    def get_y_pred(self, clf, x_test):
        y_pred = clf.predict(x_test)
        return y_pred

    def evaluate_predictions(self, y_true, y_pred):
        score = balanced_accuracy_score(y_true, y_pred)
        return score

    def evaluate_fold(self, clf, x_test):
        clf = self.move_classifier_to_cpu(clf)
        y_true = self.get_y_true(x_test)
        y_pred = self.get_y_pred(clf, x_test)
        score = self.evaluate_predictions(y_true, y_pred)
        return score

    def objective(self, trial, emotion, n_chans, sfreq, input_window_samples, n_outputs):
        params = self.suggest_params(trial)

        self.prepare_model(
            n_chans=n_chans,
            sfreq=sfreq,
            input_window_samples=input_window_samples,
            n_outputs=n_outputs,
        )

        raw_dataset = self.get_dataset(emotion)
        splits = raw_dataset.split("subject")
        subject_keys = list(splits.keys())

        scores = []

        for fold_idx, test_subject in enumerate(subject_keys):
            x_train, x_valid, x_test = self.prepare_fold_data(
                splits=splits,
                subject_keys=subject_keys,
                test_subject=test_subject,
                emotion=emotion,
            )

            clf = self.create_classifier(params, x_valid)
            clf = self.fit_classifier(clf, x_train)

            score = self.evaluate_fold(clf, x_test)
            scores.append(score)

            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    def save_best_trial(self, out_dir):
        # Add the model name to the output directory to avoid mixing JSON files for different models
        model_out_dir = os.path.join(out_dir, self.runner.model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        output = {
            "model": self.runner.model_name,
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "n_trials": len(self.study.trials),
        }

        with open(os.path.join(model_out_dir, "study_best.json"), "w") as f:
            json.dump(output, f, indent=4)

    def optimize(
            self,
            emotion,
            n_chans,
            sfreq,
            input_window_samples,
            n_outputs,
            n_trials=30,
            study_name=None,
            storage=None,
            out_dir="results_loso/optuna",
    ):
        study = self.create_study(study_name=study_name, storage=storage)

        study.optimize(
            lambda trial: self.objective(
                trial=trial,
                emotion=emotion,
                n_chans=n_chans,
                sfreq=sfreq,
                input_window_samples=input_window_samples,
                n_outputs=n_outputs,
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        self.save_best_trial(out_dir)
        return study