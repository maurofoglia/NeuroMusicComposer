"""
Main entry point for the Music Generator Application.
Handles OSC (Open Sound Control) communication, manages the application state,
and orchestrates the interaction between the LLM and Max/MSP.
"""

import time
import random
import logging
import re
from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher

from config import LLM_BACKEND, ACTIVE_MODEL, RUN_DIR, LOG_FILE, SCORE_EXPORT_PATH
from state import AppState
from llm_manager import MusicLLM
from max_communicator import MaxMSPCommunicator

# --- LOGGER CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MusicOSCApp:
    """
    Main application class that handles OSC server routing and triggers
    musical generation workflows based on incoming network messages.
    """

    def __init__(self):
        self.state = AppState()
        self.llm = MusicLLM()
        self.max_comm = MaxMSPCommunicator(ip="127.0.0.1", port_out=8000)

        logger.info(f"Experiment setup completed: {RUN_DIR}")
        logger.info(f"[{LLM_BACKEND} Backend] Initializing... Model: {ACTIVE_MODEL}")

    def generate(self, emotion):
        """
        Requests a new musical segment from the LLM based on the provided emotion,
        handles playback timing, and routes the data to Max/MSP.
        """
        self.max_comm.send_message("/emotion", emotion)

        # Request a new musical segment
        if data := self.llm.get_segment(emotion, self.state):

            # Manage timing and playback synchronization
            if self.state.next_deadline != 0 and (wait_time := self.state.next_deadline - time.time() - 0.2) > 0:
                time.sleep(wait_time)

            beat_duration = self.max_comm.send_segment(data, self.state)

            # Update the deadline for the next segment
            if self.state.next_deadline == 0:
                self.state.next_deadline = time.time() + beat_duration
            else:
                self.state.next_deadline += beat_duration

    # --- OSC HANDLERS ---

    def handle_style(self, addr, *args):
        if args:
            self.state.musical_style = str(args[0])

    def handle_custom(self, addr, *args):
        if args:
            self.state.user_custom_instructions = " ".join(str(a) for a in args)

    def handle_temperature(self, addr, *args):
        """
        Adjusts the LLM temperature and automatically maps it to a stylistic constraint
        to guide the generative creativity appropriately.
        """
        if args:
            try:
                val = max(0.0, min(1.0, float(args[0])))
                # Map incoming 0.0 - 1.0 to a narrower LLM temperature range (0.1 - 0.2)
                self.state.model_temperature = 0.1 + (val * 0.1)

                # Assign categorical creativity constraints based on the value thresholds
                if val <= 0.33:
                    self.state.creativity_musical_prompt = "STYLISTIC CONSTRAINT: Predictable."
                elif val <= 0.66:
                    self.state.creativity_musical_prompt = "STYLISTIC CONSTRAINT: Moderate variety."
                else:
                    self.state.creativity_musical_prompt = "STYLISTIC CONSTRAINT: Expressive, complex."
            except ValueError:
                pass

    def handle_refinement(self, addr, *args):
        if args:
            self.state.refinement_level = int(args[0])

    def handle_num_voices(self, addr, *args):
        if args:
            self.state.num_voices = max(1, min(4, int(args[0])))

    def handle_instrument(self, addr, *args):
        # Extract the voice index from the OSC address and update the corresponding instrument
        if args and (m := re.search(r'/voice(\d+)_inst', addr)):
            voice_index = int(m.group(1))
            self.state.active_instruments[voice_index] = " ".join(str(a) for a in args)

    def handle_re2(self, addr, *args):
        if args:
            self.state.use_re2 = bool(int(args[0]))

    def handle_few_shot(self, addr, *args):
        if args:
            self.state.use_few_shot = bool(int(args[0]))

    def handle_start(self, addr, *args):
        """
        Initializes a new generation session, clears previous data, and kicks off
        a sequence of randomly chosen emotions.
        """
        self.state.reset_for_new_sequence()

        for msg in ["/clear", "/clear_diary"]:
            self.max_comm.send_message(msg, "bang")

        time.sleep(0.05)
        self.llm.initialize_session(self.state)

        # Generate 4 random emotive sequences
        for emotion in random.choices(["Excited", "Relaxed", "Sad", "Angry"], k=4):
            self.generate(emotion)

        time.sleep(2)

        # Export the resulting score automatically
        self.max_comm.send_message("/export_score", str(SCORE_EXPORT_PATH.resolve()))

    def run(self):
        """
        Starts the main OSC server loop to listen for incoming configuration and trigger messages.
        """
        logger.info("=== STARTING MUSIC GENERATOR (LISTENING) ===")
        dispatcher = Dispatcher()

        # Route definitions mapping OSC addresses to class methods
        routes = {
            "/start": self.handle_start,
            "/temperature": self.handle_temperature,
            "/custom_prompt": self.handle_custom,
            "/style": self.handle_style,
            "/refinement_level": self.handle_refinement,
            "/use_re2": self.handle_re2,
            "/use_few_shot": self.handle_few_shot,
            "/numvoices_set": self.handle_num_voices,
            "/voice*_inst": self.handle_instrument
        }

        for route, func in routes.items():
            dispatcher.map(route, func)

        # Log an event if an unmapped OSC address is received
        dispatcher.set_default_handler(
            lambda address, *args: logger.warning(f"Unknown OSC routing -> Addr: '{address}', Args: {args}")
        )

        try:
            # Bind to localhost on port 8001
            server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 8001), dispatcher)
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down the server gracefully.")


if __name__ == "__main__":
    app = MusicOSCApp()
    app.run()