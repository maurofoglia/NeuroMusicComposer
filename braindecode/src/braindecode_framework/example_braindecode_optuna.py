from src.braindecode_framework.braindecode_framework import BrainDecodeFramework
from src.braindecode_framework.braindecode_optuna import BraindecodeOptuna


class RunBraindecodeOptuna:
    def __init__(self, model_name="CTNet", model_kwargs=None):
        self.model_name = model_name

        # Initialize the framework by passing the chosen model and any extra parameters
        self.framework = BrainDecodeFramework(model_name=self.model_name, model_kwargs=model_kwargs)
        self.tuner = BraindecodeOptuna(self.framework)

    def prepare_data(self):
        """
        Here you must call your actual dataset loading steps,
        BaseConcatDataset creation, and preprocessing before tuning.
        """
        pass

    def run(self, emotion="excited", n_trials=30):
        self.prepare_data()

        # Make the study name and folder dynamic as well!
        study_name = f"{self.model_name.lower()}_{emotion}_optuna"
        out_dir = f"results_loso/optuna/{self.model_name}/{emotion}"

        print(f"========== Starting Pipeline for Model: {self.model_name} ==========")

        study = self.tuner.optimize(
            emotion=emotion,
            n_chans=4,
            sfreq=128,
            input_window_samples=512,
            n_outputs=2,
            n_trials=n_trials,
            study_name=study_name,
            storage=None,
            out_dir=out_dir,
        )

        print(f"\n[{self.model_name}] best_value:", study.best_value)
        print(f"[{self.model_name}] best_params:", study.best_params)


if __name__ == "__main__":
    # =====================================================================
    # Change this string to try any braindecode model!
    # Supported examples: "CTNet", "EEGNetv4", "ShallowFBCSPNet", "Deep4Net"
    # =====================================================================
    MODEL_NAME = "EEGNetv4"

    # If the model requires specific parameters that you are not passing via Optuna,
    # you can put them in this dictionary. Otherwise, leave it empty.
    MODEL_KWARGS = {
        # Example: "pool_mode": "mean", "F1": 8
    }

    example = RunBraindecodeOptuna(model_name=MODEL_NAME, model_kwargs=MODEL_KWARGS)

    # Start the run
    example.run(emotion="excited", n_trials=30)