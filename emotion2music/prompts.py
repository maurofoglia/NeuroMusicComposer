"""
Prompt Generation Module.
Responsible for building and formatting the context, constraints, and instructions
sent to the LLMs for both base generation and iterative refinement passes.
"""

from instrument_info import INSTRUMENT_INFO

JSON_FORMAT_EXAMPLES = {
    1: """
**EXPECTED JSON FORMAT EXAMPLE (N = 1, Solo, 1 Measure)**
{
    "composer_diary": "1. Configuration: N=1 voice requested. Output requires 1 top-level block. 2. Emotion: Calm. 3. Role: Flute melody. 4. Rhythm Check: V1M1 [1/2 1/4 1/4] = 1. 5. Pitch Plan: Bounds respected.",
    "measures": "[ [[4 4] [72]] ]",
    "pitches": "[ [A4 D5 F5] ]",
    "durations": "[ [1/2 1/4 1/4] ]",
    "velocities": "[ [70 75 80] ]",
    "ties": "[ [0 0 0] ]",
    "articulations": "[ [ [ [] ] [ [tenuto] ] [ [] ] ] ]"
}""",
    2: """
**EXPECTED JSON FORMAT EXAMPLE (N = 2, Duet, 2 Measures)**
{
    "composer_diary": "1. Configuration: N=2 voices. Output requires 2 top-level blocks. 2. Emotion: Melancholy. 3. Roles: V1 Violin, V2 Cello. 4. Rhythm Check: V1M1 [1/4 1/4 1/4 1/4]=1, V1M2 [1/2 1/2]=1. V2M1 [1/2 1/4 1/4]=1, V2M2 [1/2 1/2]=1. 5. Pitch Plan: Bounds respected.",
    "measures": "[ [[4 4] [72]] [[4 4] [72]] ]",
    "pitches": "[ [A4 D5 F5 E5] [G5 F5] ] [ [D2 A2 D3] [C#3 A2] ]",
    "durations": "[ [1/4 1/4 1/4 1/4] [1/2 1/2] ] [ [1/2 1/4 1/4] [1/2 1/2] ]",
    "velocities": "[ [70 75 80 75] [85 75] ] [ [60 55 55] [65 50] ]",
    "ties": "[ [0 0 0 0] [0 0] ] [ [0 0 0] [0 0] ]",
    "articulations": "[ [ [ [] ] [ [] ] [ [] ] [ [tenuto] ] ] [ [ [tenuto] ] [ [] ] ] ] [ [ [ [tenuto] ] [ [] ] [ [] ] ] [ [ [accent] ] [ [] ] ] ]"
}""",
    3: """
**EXPECTED JSON FORMAT EXAMPLE (N = 3, Trio, 1 Measure)**
{
    "composer_diary": "1. Configuration: N=3 voices. Output requires 3 top-level blocks. 2. Emotion: Joyful. 3. Roles: V1 Oboe, V2 Clarinet, V3 Bassoon. 4. Rhythm Check: V1M1 [1/4 1/4 1/2]=1. V2M1 [1/2 1/2]=1. V3M1 [1]=1. 5. Pitch Plan: Bounds respected.",
    "measures": "[ [[4 4] [72]] ]",
    "pitches": "[ [C5 E5 G5] ] [ [G3 B3] ] [ [C2] ]",
    "durations": "[ [1/4 1/4 1/2] ] [ [1/2 1/2] ] [ [1] ]",
    "velocities": "[ [80 85 90] ] [ [70 75] ] [ [80] ]",
    "ties": "[ [0 0 0] ] [ [0 0] ] [ [0] ]",
    "articulations": "[ [ [ [staccato] ] [ [staccato] ] [ [accent] ] ] ] [ [ [ [] ] [ [] ] ] ] [ [ [ [tenuto] ] ] ]"
}""",
    4: """
**EXPECTED JSON FORMAT EXAMPLE (N = 4, Quartet, 1 Measure)**
{
    "composer_diary": "1. Configuration: N=4 voices. Output requires 4 top-level blocks. 2. Emotion: Solemn. 3. Roles: String Quartet (V1, V2, V3, V4). 4. Rhythm Check: V1M1 [1/4 1/4 1/2]=1. V2M1 [1/2 1/4 1/4]=1. V3M1 [1/2 1/2]=1. V4M1 [1]=1. 5. Pitch Plan: Bounds respected.",
    "measures": "[ [[4 4] [72]] ]",
    "pitches": "[ [C5 E5 G5] ] [ [G4 C5 E5] ] [ [E4 G4] ] [ [C3] ]",
    "durations": "[ [1/4 1/4 1/2] ] [ [1/2 1/4 1/4] ] [ [1/2 1/2] ] [ [1] ]",
    "velocities": "[ [75 80 85] ] [ [70 75 70] ] [ [65 70] ] [ [80] ]",
    "ties": "[ [0 0 0] ] [ [0 0 0] ] [ [0 0] ] [ [0] ]",
    "articulations": "[ [ [ [tenuto] ] [ [] ] [ [] ] ] ] [ [ [ [] ] [ [] ] [ [] ] ] ] [ [ [ [tenuto] ] [ [] ] ] ] [ [ [ [accent] ] ] ]"
}"""
}


def build_generation_prompt(emotion, num_measures, num_voices, active_instruments, current_bpm,
                            musical_style, creativity_prompt, user_custom, use_re2, use_few_shot):
    """Constructs the complete base prompt for the LLM based on current configurations."""

    instrument_constraints = "### CRITICAL INSTRUMENT CONSTRAINTS (DO NOT DRIFT) ###\n"
    instrument_constraints += f"You MUST generate EXACTLY {num_voices} musical voices. Depth of your LLLL arrays must correspond to {num_voices}.\n\n"

    for i in range(1, num_voices + 1):
        instrument_name = active_instruments.get(i, "1 Piano 1")
        info = INSTRUMENT_INFO.get(instrument_name, INSTRUMENT_INFO["1 Piano 1"])
        instrument_constraints += f"- VOICE {i} ({instrument_name}):\n  * EXACT RANGE ALLOWED: {info['range']}.\n  * IDIOM: {info['idiom']}\n\n"

    if use_few_shot:
        targeted_example = JSON_FORMAT_EXAMPLES.get(num_voices, JSON_FORMAT_EXAMPLES.get(2, ""))
        technical_context = f"""{instrument_constraints}\n### CRITICAL: JSON FORMAT EXAMPLE FOR EXACTLY {num_voices} VOICES ###\n{targeted_example}\n"""
        log_prompt_type = "Few-Shot"
    else:
        technical_context = f"{instrument_constraints}\n"
        log_prompt_type = "Zero-Shot"

    musical_query = f"Generate exactly {num_measures} measures for {num_voices} voices. The requested emotion is: {emotion}."

    if current_bpm is not None:
        musical_query += f" IMPERATIVE: Strictly maintain the BPM at {current_bpm} (the third value in the 'measures' arrays)."
    else:
        musical_query += f" ATTENTION: Freely choose an initial BPM creatively suited for the emotion '{emotion}' and insert it as the third value in the 'measures' array."

    if musical_style.lower() != "auto" and musical_style.strip():
        musical_query += f" Explicitly use this MUSICAL STYLE: {musical_style}."

    if creativity_prompt:
        musical_query += f" {creativity_prompt}"

    if user_custom.strip():
        musical_query += f" [ADDITIONAL INSTRUCTIONS: {user_custom}]"

    # Add reasoning tags if the use_re2 flag is active
    if use_re2:
        prompt_base = f"{technical_context}\nQ: {musical_query}\n\nRead the question again: {musical_query}\n\nGenerate exclusively the raw JSON."
    else:
        prompt_base = f"{technical_context}\nTASK: {musical_query}\n\nGenerate exclusively the raw JSON."

    return prompt_base, log_prompt_type


def generate_auto_refine_prompt(emotion, current_json, num_voices, active_instruments, locked_bpm=None):
    """Constructs a prompt for a single-pass automatic refinement of generated music."""

    range_criteria = ""
    for i in range(1, num_voices + 1):
        instrument_name = active_instruments.get(i, "1 Piano 1")
        info = INSTRUMENT_INFO.get(instrument_name, INSTRUMENT_INFO["1 Piano 1"])
        range_criteria += f"Voice {i} ({instrument_name}) MUST strictly respect the range {info['range']}. "

    bpm_rule = ""
    if locked_bpm is not None:
        bpm_rule = f"\n5. BPM Consistency: The BPM (the third number in the 'measures' array) must be EXACTLY {locked_bpm} in all measures."

    return f"""You are an expert music proofreader. Analyze and directly correct the following JSON musical score generated for the emotion '{emotion}'.
Strictly apply the following rules and return ONLY the corrected JSON:
1. Rhythmic Math: The sum of the durations in every single measure MUST correspond exactly to the time signature for ALL voices.
2. Voice Ranges: {range_criteria}
3. Harmony: Ensure that voices form consonant intervals on downbeats.
4. JSON Validity: The LLLL arrays for the {num_voices} voices must have consistent depths and lengths.{bpm_rule}

CURRENT JSON:
{current_json}

Autonomously resolve any issues (out-of-range notes, rhythmic mathematical errors).
Return EXCLUSIVELY a valid raw JSON object starting with {{ and ending with }}. No markdown, no conversational text.
CORRECTED JSON:"""


def generate_feedback_prompt(emotion, current_json, num_voices, active_instruments, locked_bpm=None):
    """Constructs a prompt to critique the generated score and provide strict feedback."""

    range_criteria = ""
    for i in range(1, num_voices + 1):
        instrument_name = active_instruments.get(i, "1 Piano 1")
        info = INSTRUMENT_INFO.get(instrument_name, INSTRUMENT_INFO["1 Piano 1"])
        range_criteria += f"Does Voice {i} ({instrument_name}) strictly respect the range {info['range']}? "

    bpm_rule = ""
    if locked_bpm is not None:
        bpm_rule = f"\n5. BPM Consistency: The BPM (the third number in the 'measures' array) must be EXACTLY {locked_bpm} in all measures."

    return f"""Critically analyze this JSON musical score generated for the emotion '{emotion}'.
Act as a strict music theory and data syntax reviewer. Perform a step-by-step check on the following criteria:
1. Rhythmic Math: Does the sum of the durations in every single measure correspond exactly to the time signature for ALL voices?
2. Voice Ranges: {range_criteria}
3. Harmony & Consonance: On downbeats, do the voices form consonant intervals? Does the final note resolve to the Tonic?
4. JSON Validity: Do the arrays of pitches, durations, velocities, and articulations all share the same length and LLLL depth corresponding to {num_voices} voices?{bpm_rule}

If you DO NOT find any errors across all criteria, write EXACTLY AND ONLY this phrase: "NO ERROR".

CRITIQUE EXAMPLE (Few-Shot Demonstration):
JSON: {{"durations": "[[1/8 1/8 1/4]] [[1/4 1/4]]"}}
Critique:
- Rhythm: Error. V1 sums to 2/4 (missing 2/4 to complete). V2 sums to 2/4. V1 uses eighth notes, but V2 is too static.
- Range: No errors detected in this portion.
- Required Action: Complete the durations in all voices to reach 4/4 using mixed rhythms (e.g., 1/8, 1/16, dotted notes).

REAL TASK:
JSON: {current_json}
Critique:"""


def generate_refine_prompt(current_json, feedback):
    """Constructs a prompt to apply the previously generated feedback to the JSON score."""

    return f"""You are a JSON score corrector for bach.score in Max/MSP.
Your task is to apply EXACTLY the corrections requested in the Critique to the provided JSON.

CORRECTION EXAMPLE (One-Shot Demonstration):
CURRENT JSON: {{"pitches": "[[C4 C7]] [[A1]]", "durations": "[[1/4 1/2]] [[1/2 1/2]]"}}
CRITIQUE: "In Voice 1, C7 is out of range. In Voice 2, A1 is out of range. Furthermore, V1 sums to 3/4."
CORRECTED JSON: {{"pitches": "[[C4 C6 D6 C6]] [[C2 G2 C3 E3]]", "durations": "[[1/8 1/8 1/4 1/2]] [[1/8 1/8 1/8 1/8 1/2]]"}}

REAL TASK:
CURRENT JSON: {current_json}
CRITIQUE: {feedback}

Apply the corrections and resolve the problems outlined in the feedback.
Return EXCLUSIVELY a valid raw JSON object starting with {{ and ending with }}. No conversational text, no markdown (do not use ```json).
CORRECTED JSON:"""