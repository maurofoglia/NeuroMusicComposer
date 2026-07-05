"""
Configuration Module.
Handles LLM backend selection, model configurations, and dynamically sets up
the directory structure and file paths for experiment logging and exports.
"""

from pathlib import Path
from datetime import datetime

# --- BACKEND AND MODEL SELECTION ---
LLM_BACKEND = 'Claude'

MODEL_CONFIG = {
    'Gemini': {
        'active_model': 'gemini-3.1-flash-lite',
        'available_models': [
            'gemini-3.1-flash-lite', 'gemini-3.5-flash', 'gemini-2.5-flash',
            'gemini-3.1-pro-preview', 'gemini-2.5-pro', 'gemini-pro-latest', 'gemini-flash-latest'
        ]
    },
    'Ollama': {
        'active_model': 'gemma2:2b',
        'available_models': ['qwen2.5-coder:1.5b', 'gemma2:2b', 'llama3']
    },
    'Groq': {
        'active_model': 'openai/gpt-oss-120b',
        'available_models': ['openai/gpt-oss-120b', 'llama-3.3-70b-versatile', 'qwen/qwen3-32b']
    },
    'GitHub': {
        'active_model': 'gpt-5-mini',
        'available_models': [
            "gpt-5", "gpt-5-chat-preview", "gpt-5-mini", "gpt-5-nano", "o4-mini", "o3", "o3-mini",
            "o1", "o1-preview", "o1-mini", "gpt-4o", "gpt-4o-mini", "text-embedding-3-large",
            "text-embedding-3-small", "Phi-4", "Phi-4-mini-instruct", "Phi-4-mini-reasoning",
            "Phi-4-multimodal-instruct", "Phi-4-reasoning", "Llama-3-3-70B-Instruct",
            "Meta-Llama-3-1-405B-Instruct", "Llama-3-2-11B-Vision-Instruct", "mistral-small-2503",
            "mistral-medium-2505", "Ministral-3B", "Codestral-2501", "cohere-command-a",
            "Cohere-command-r-plus-08-2024", "AI21-Jamba-1-5-Large"
        ]
    },
    'Claude': {
        'active_model': 'claude-haiku-4-5-20251001',
        'available_models': [
            'claude-fable-5', 'claude-opus-4-8', 'claude-sonnet-4-6', 'claude-haiku-4-5-20251001'
        ]
    },
    'OpenRouter': {
        'active_model': 'openai/gpt-oss-120b:free',
        'available_models': [
            'openrouter/free', 'deepseek/deepseek-r1:free', 'deepseek/deepseek-v4-flash:free',
            'deepseek/deepseek-chat-v3-0324:free', 'meta-llama/llama-4-maverick:free',
            'meta-llama/llama-4-scout:free', 'openai/gpt-oss-120b:free', 'openai/gpt-oss-20b:free',
            'qwen/qwen3-next-80b-a3b-instruct:free', 'qwen/qwen2.5-coder-32b-instruct:free',
            'nvidia/nemotron-3-super-120b-a12b:free', 'google/gemma-4-31b:free',
            'moonshotai/kimi-vl-a3b-thinking:free', 'anthropic/claude-3.5-sonnet',
            'google/gemini-2.5-flash', 'meta-llama/llama-3.3-70b-instruct', 'openai/gpt-4o',
            'mistralai/mistral-large-2411', 'deepseek/deepseek-chat'
        ]
    }
}

OLLAMA_MODEL = MODEL_CONFIG['Ollama']['active_model']
GROQ_MODEL = MODEL_CONFIG['Groq']['active_model']
GITHUB_MODEL = MODEL_CONFIG['GitHub']['active_model']
GEMINI_MODEL = MODEL_CONFIG['Gemini']['active_model']
CLAUDE_MODEL = MODEL_CONFIG['Claude']['active_model']
OPENROUTER_MODEL = MODEL_CONFIG['OpenRouter']['active_model']

ACTIVE_MODEL = MODEL_CONFIG[LLM_BACKEND]['active_model']

# --- EXPERIMENT DIRECTORY AND FILE PATH SETUP ---
# Sanitize the model name for file system compatibility
SAFE_MODEL_NAME = ACTIVE_MODEL.replace('/', '_').replace(':', '_')
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_NAME = f"{LLM_BACKEND}_{SAFE_MODEL_NAME}_{TIMESTAMP}"

# Create the unique directory for this specific run
RUN_DIR = Path("experiments") / BASE_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Define absolute paths for all output files
LOG_FILE = RUN_DIR / f"{BASE_NAME}.log"
CSV_FILE = RUN_DIR / f"{BASE_NAME}.csv"
XLSX_FILE = RUN_DIR / f"{BASE_NAME}.xlsx"
DIARY_FILE = RUN_DIR / f"{BASE_NAME}_diaries.txt"
JSON_FILE = RUN_DIR / f"{BASE_NAME}_jsons.txt"
SCORE_EXPORT_PATH = RUN_DIR / f"{BASE_NAME}_score.xml"