"""
LLM Manager Module.
Handles API client initialization, chat session management, prompt execution,
and metric logging for various LLM backends (Gemini, Groq, Claude, OpenAI, Ollama).
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from json_repair import repair_json

import ollama
import anthropic
from google import genai
from google.genai import types
from groq import Groq
from openai import OpenAI

from config import (
    LLM_BACKEND, GEMINI_MODEL, GROQ_MODEL, GITHUB_MODEL,
    CLAUDE_MODEL, OPENROUTER_MODEL, OLLAMA_MODEL, ACTIVE_MODEL,
    CSV_FILE, XLSX_FILE, DIARY_FILE, JSON_FILE
)

# Note: These function names have been translated.
# They will need to match the updated utils.py and prompts.py files.
from utils import extract_bpm, export_logs_and_metrics
from prompts import (
    build_generation_prompt,
    generate_auto_refine_prompt,
    generate_feedback_prompt,
    generate_refine_prompt
)

logger = logging.getLogger(__name__)


class MusicLLM:
    """
    Manages the communication with the selected Large Language Model, handling
    generation, refinement loops, and API configuration.
    """

    def __init__(self):
        self.client = self.groq_client = self.github_client = self.claude_client = self.openrouter_client = None

        # Initialize system instructions
        sys_path = Path("system_instruct_new.txt")
        if sys_path.exists():
            self.system_instructions = sys_path.read_text(encoding="utf-8")
        else:
            self.system_instructions = "Generate {NUM_MEASURES} measures of music in JSON format."

        self._init_clients()

    def _init_clients(self):
        """Loads API keys from text files and initializes the respective LLM clients."""

        def load_key(file_path):
            p = Path(file_path)
            if p.exists():
                return p.read_text(encoding="utf-8").strip()
            logger.warning(f"WARNING: API key file {file_path} not found.")
            return None

        if LLM_BACKEND == 'Gemini':
            if key := load_key("api_key_gemini.txt"):
                self.client = genai.Client(api_key=key)
        elif LLM_BACKEND == 'Groq':
            if key := load_key("api_key_groq.txt"):
                self.groq_client = Groq(api_key=key)
        elif LLM_BACKEND == 'GitHub':
            if key := load_key("api_key_github.txt"):
                self.github_client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=key)
        elif LLM_BACKEND == 'Claude':
            if key := load_key("api_key_claude.txt"):
                self.claude_client = anthropic.Anthropic(api_key=key)
        elif LLM_BACKEND == 'OpenRouter':
            if key := load_key("api_key_openrouter.txt"):
                self.openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        elif LLM_BACKEND == 'Ollama':
            logger.info("Ollama client configured for local execution.")

    def initialize_session(self, state):
        """Prepares the chat context and system instructions for a new generation sequence."""
        logger.info(f"Initializing session for {LLM_BACKEND} with temperature: {state.model_temperature}")
        instructions = self.system_instructions.replace("{NUM_MEASURES}", str(state.num_measures))

        if LLM_BACKEND == 'Gemini' and self.client:
            self.chat_session = self.client.chats.create(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=instructions,
                    response_mime_type="application/json",
                    temperature=state.model_temperature
                )
            )
            state.chat_session = self.chat_session
        elif LLM_BACKEND in ['Groq', 'GitHub', 'OpenRouter', 'Ollama']:
            setattr(state, f"{LLM_BACKEND.lower()}_chat_history", [{'role': 'system', 'content': instructions}])
        elif LLM_BACKEND == 'Claude':
            state.claude_chat_history = []

    def execute_pass(self, prompt, state, is_refinement=False, json_format=True):
        """
        Sends a prompt to the active LLM backend and processes the response.
        Automatically lowers the temperature slightly during refinement passes.
        """
        current_temp = state.model_temperature if not is_refinement else max(0.05, state.model_temperature - 0.05)
        messages_to_send = []

        if LLM_BACKEND != 'Gemini':
            messages_to_send = list(getattr(state, f"{LLM_BACKEND.lower()}_chat_history", []))
        if LLM_BACKEND in ['Groq', 'GitHub', 'Ollama', 'Claude', 'OpenRouter']:
            messages_to_send.append({'role': 'user', 'content': prompt})

        start_time = time.perf_counter()

        try:
            if LLM_BACKEND == 'Gemini':
                instructions = self.system_instructions.replace("{NUM_MEASURES}", str(state.num_measures))
                if is_refinement:
                    resp = self.client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=instructions,
                            response_mime_type="application/json" if json_format else "text/plain",
                            temperature=current_temp
                        )
                    )
                else:
                    resp = state.chat_session.send_message(prompt)
                text_response = resp.text

            elif LLM_BACKEND == 'Groq':
                response = self.groq_client.chat.completions.create(
                    messages=messages_to_send,
                    model=GROQ_MODEL,
                    temperature=current_temp,
                    response_format={"type": "json_object"} if json_format else None
                )
                text_response = response.choices[0].message.content.strip()

            elif LLM_BACKEND == 'GitHub':
                response = self.github_client.chat.completions.create(
                    messages=messages_to_send,
                    model=GITHUB_MODEL,
                    temperature=current_temp,
                    response_format={"type": "json_object"} if json_format else None
                )
                text_response = response.choices[0].message.content.strip()

            elif LLM_BACKEND == 'OpenRouter':
                response = self.openrouter_client.chat.completions.create(
                    messages=messages_to_send,
                    model=OPENROUTER_MODEL,
                    temperature=current_temp,
                    response_format={"type": "json_object"} if json_format else None
                )
                text_response = response.choices[0].message.content.strip()

            elif LLM_BACKEND == 'Ollama':
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=messages_to_send,
                    format="json" if json_format else "",
                    options={'temperature': current_temp}
                )
                text_response = response['message']['content'].strip()
                if json_format:
                    text_response = text_response.replace("```json", "").replace("```", "").strip()

            elif LLM_BACKEND == 'Claude':
                instructions = self.system_instructions.replace("{NUM_MEASURES}", str(state.num_measures))
                kwargs = {
                    "model": CLAUDE_MODEL,
                    "max_tokens": 4096,
                    "system": instructions,
                    "messages": messages_to_send
                }

                # Temperature is not supported by Opus/Fable models on Claude in this specific setup
                if "fable" not in CLAUDE_MODEL.lower() and "opus" not in CLAUDE_MODEL.lower():
                    kwargs["temperature"] = current_temp

                response = self.claude_client.messages.create(**kwargs)
                text_response = "".join([b.text for b in response.content if hasattr(b, 'text') and b.text]).strip()

                # Extract JSON block if present
                if json_format and '{' in text_response:
                    text_response = text_response[text_response.find('{'):text_response.rfind('}') + 1]

            pass_type = "Refinement" if is_refinement else "Base Generation"
            logger.info(f"[THESIS-METRIC] Type='{pass_type}' | API_Latency={time.perf_counter() - start_time:.3f}s")

            if json_format:
                try:
                    return json.loads(text_response)
                except json.JSONDecodeError:
                    return json.loads(repair_json(text_response))

            return text_response

        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return None

    def get_segment(self, emotion, state):
        """
        Orchestrates the generation of a musical segment based on the emotion.
        Handles the base generation and subsequent evaluation/refinement loops if configured.
        """
        logger.info(
            f"Request -> Emotion: {emotion} | Voices: {state.num_voices} | "
            f"Refine Level: {state.refinement_level} | RE2: {state.use_re2} | Few-Shot: {state.use_few_shot}"
        )

        prompt_base, log_prompt_type = build_generation_prompt(
            emotion, state.num_measures, state.num_voices, state.active_instruments,
            state.current_bpm, state.musical_style, state.creativity_musical_prompt,
            state.user_custom_instructions, state.use_re2, state.use_few_shot
        )

        task_start_time = time.perf_counter()

        # Initial base generation
        musical_data = self.execute_pass(prompt_base, state, is_refinement=False, json_format=True)
        if not musical_data:
            return None

        executed_iterations = 0

        # Handle auto-refinement (Level 1)
        if state.refinement_level == 1:
            refined_data = self.execute_pass(
                generate_auto_refine_prompt(
                    emotion, json.dumps(musical_data), state.num_voices,
                    state.active_instruments, state.current_bpm
                ),
                state,
                is_refinement=True,
                json_format=True
            )
            if refined_data:
                musical_data, executed_iterations = refined_data, 1

        # Handle feedback-based multi-step refinement (Level 2+)
        elif state.refinement_level >= 2:
            feedback = self.execute_pass(
                generate_feedback_prompt(
                    emotion, json.dumps(musical_data), state.num_voices,
                    state.active_instruments, state.current_bpm
                ),
                state,
                is_refinement=True,
                json_format=False
            )

            if feedback:
                if "NO ERROR" not in feedback.upper():
                    refined_data = self.execute_pass(
                        generate_refine_prompt(json.dumps(musical_data), feedback),
                        state,
                        is_refinement=True,
                        json_format=True
                    )
                    if refined_data:
                        musical_data, executed_iterations = refined_data, 2
                else:
                    executed_iterations = 1

        total_latency = time.perf_counter() - task_start_time

        # Compile metrics for logging
        metric_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Task": "Completed",
            "Backend": LLM_BACKEND,
            "Model": ACTIVE_MODEL,
            "Temperature": state.model_temperature,
            "Emotion": emotion,
            "Style": state.musical_style,
            "BPM": state.current_bpm or 'Auto',
            "Voices": state.num_voices,
            "Measures": state.num_measures,
            "Prompt_Type": log_prompt_type,
            "Use_RE2": state.use_re2,
            "Total_Latency (s)": round(total_latency, 3),
            "Latency_Per_Measure (s)": round(total_latency / max(1, state.num_measures), 3),
            "Refine_Cycles": executed_iterations
        }

        export_logs_and_metrics(musical_data, metric_data, CSV_FILE, XLSX_FILE, DIARY_FILE, JSON_FILE, logger)

        # Extrapolate BPM if auto-detecting
        if state.current_bpm is None and "measures" in musical_data:
            if detected_bpm := extract_bpm(musical_data["measures"]):
                state.current_bpm = int(detected_bpm) if float(detected_bpm).is_integer() else float(detected_bpm)

        # Update chat history context for text-based conversational models
        if LLM_BACKEND != 'Gemini':
            history_attr = f"{LLM_BACKEND.lower()}_chat_history"
            history = getattr(state, history_attr, [])
            history.extend([
                {'role': 'user', 'content': prompt_base},
                {'role': 'assistant', 'content': json.dumps(musical_data)}
            ])
            # Keep history truncated to avoid exceeding token limits
            if len(history) > 9:
                setattr(state, history_attr, [history[0]] + history[-8:])

        return musical_data