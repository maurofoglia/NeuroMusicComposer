"""
Application State Management.
Holds the configuration, execution memory, and LLM context for the current session.
"""


class AppState:
    """
    Maintains the global state of the application, including generative parameters,
    instrument routing, the musical score being built, and LLM chat histories.
    """

    def __init__(self):
        # --- Generative Parameters ---
        self.model_temperature = 0.1
        self.creativity_musical_prompt = ""
        self.user_custom_instructions = ""
        self.musical_style = "auto"
        self.refinement_level = 0
        self.num_measures = 2  # Translated from num_battute

        self.current_bpm = None
        self.use_re2 = True
        self.use_few_shot = True

        # --- Instrument Configuration ---
        self.num_voices = 2
        self.active_instruments = {
            1: "1 Piano 1",
            2: "34 Fingered Bs."
        }

        # --- Execution Memory (Score under construction) ---
        self.total_measures = 0
        self.next_deadline = 0

        # Initialize an empty score dictionary with required musical attributes
        self.full_score = {
            k: None for k in ["measures", "pitches", "durations", "velocities", "ties", "articulations"]
        }

        # --- LLM Context & Session Memory ---
        self.chat_session = None
        self.groq_chat_history = []
        self.github_chat_history = []
        self.claude_chat_history = []
        self.openrouter_chat_history = []
        self.ollama_chat_history = []

    def reset_for_new_sequence(self):
        """
        Resets the state memory to prepare for a completely new generation sequence,
        clearing the score, deadlines, and all LLM chat histories.
        """
        self.total_measures = 0
        self.next_deadline = 0
        self.current_bpm = None
        self.chat_session = None

        # Reset the score dictionary values to None
        self.full_score = {k: None for k in self.full_score}

        # Clear all specific LLM histories
        self.groq_chat_history = []
        self.github_chat_history = []
        self.claude_chat_history = []
        self.openrouter_chat_history = []
        self.ollama_chat_history = []