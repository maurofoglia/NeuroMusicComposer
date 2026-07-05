"""
Max/MSP Communicator.
Handles the formatting and transmission of musical data, instrument configurations,
and playback controls to Max/MSP via Open Sound Control (OSC).
"""

import time
from pythonosc import udp_client

# Note: These function names have been translated to English.
# They will need to be updated accordingly in your utils.py file.
from utils import (
    concatenate_lists,
    transpose_pitch_list,
    calculate_total_duration,
    prepare_osc_list
)
from instrument_info import INSTRUMENT_INFO


class MaxMSPCommunicator:
    """
    Manages the UDP client connection and OSC message routing to Max/MSP.
    """

    def __init__(self, ip="127.0.0.1", port_out=8000):
        self.osc_out = udp_client.SimpleUDPClient(ip, port_out)

    def send_message(self, address, data):
        """
        Helper method to send direct OSC messages.
        """
        self.osc_out.send_message(address, data)

    def send_segment(self, musical_data, state):
        """
        Formats the generated musical segment, updates the global score state,
        and sends the necessary setup and playback arrays to Max/MSP.
        """
        if "composer_diary" in musical_data:
            self.osc_out.send_message("/diary", f"--- NEW MEASURE ---\n{musical_data['composer_diary']}\n\n")

        # Update the global score with the new segment data
        for key in ["measures", "pitches", "durations", "velocities", "ties", "articulations"]:
            if key in musical_data:
                # Sanitize the string to remove unwanted characters
                val = str(musical_data[key]).replace("'", "").replace('"', "").replace(",", "")

                # Format articulations specifically
                if key == "articulations" and "articulations" in val:
                    val = val.replace("[articulations", "").strip()[:-1].strip()

                state.full_score[key] = concatenate_lists(state.full_score[key], val)

        starting_measure = state.total_measures + 1
        state.total_measures += state.num_measures

        # Clear previous data in Max/MSP buffers before sending new data
        self.osc_out.send_message("/clear", "bang")
        time.sleep(0.05)

        # Prepare instrument setups (clefs, names, MIDI program numbers)
        clefs, names, programs = [], [], []

        for i in range(1, state.num_voices + 1):
            instrument_name = state.active_instruments.get(i, "1 Piano 1")

            # Fetch clef from the database; fallback to treble (G) piano
            clefs.append(INSTRUMENT_INFO.get(instrument_name, INSTRUMENT_INFO["1 Piano 1"])["clef"])

            # Extract instrument name without the leading MIDI number
            names.append(" ".join(instrument_name.split(" ")[1:]) if " " in instrument_name else instrument_name)

            # Extract the MIDI program number
            try:
                programs.append(int(instrument_name.split(" ")[0]))
            except ValueError:
                programs.append(1)

        # Send configuration metadata to Max/MSP
        self.osc_out.send_message("/numvoices", state.num_voices)

        setup_addresses = ["/clefs", "/voicenames", "/midiprograms", "/keys"]
        setup_data = [clefs, names, programs, ["Gm"] * state.num_voices]

        for osc_addr, data in zip(setup_addresses, setup_data):
            self.osc_out.send_message(osc_addr, data)

        # Send core musical arrays
        for msg in ["measures", "pitches", "durations", "velocities", "ties"]:
            if msg == "pitches" and state.full_score["pitches"]:
                # Transpose pitches by +2 semitones before sending
                transposed_notes = transpose_pitch_list(state.full_score["pitches"], 2)
                self.osc_out.send_message(f"/{msg}", prepare_osc_list(transposed_notes))
            else:
                self.osc_out.send_message(f"/{msg}", prepare_osc_list(state.full_score[msg]))
            time.sleep(0.01)

        # Handle articulations separately if they exist
        if state.full_score["articulations"]:
            formatted_articulations = f"[articulations {state.full_score['articulations']} ]"
            self.osc_out.send_message("/articulations", prepare_osc_list(formatted_articulations))

        # Trigger playback in Max/MSP
        self.osc_out.send_message("/bang", "bang")
        time.sleep(0.05)
        self.osc_out.send_message("/startMeasure", starting_measure)

        # Calculate and return the playback duration of the current segment
        return calculate_total_duration(musical_data.get("measures", ""))