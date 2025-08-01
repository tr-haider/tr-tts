import os
import re
import numpy as np
import soundfile as sf
import streamlit as st
from kokoro import KPipeline
from typing import Tuple, List

import spacy
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Constants
audios_directory = './audio/'
os.makedirs(audios_directory, exist_ok=True)

DEFAULT_LANG_CODE = 'a'
VOICE_POOL = ['af_heart', 'am_michael']

def clean_text(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def validate_input_text(text: str) -> Tuple[bool, str]:
    """
    Validates the input text for correct dialogue formatting.
    """
    if not text.strip():
        return False, "Input text cannot be empty."

    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    speaker_line_pattern = re.compile(r"^[A-Za-z_][a-zA-Z0-9_]*:\s.+$")

    for line in lines:
        if not speaker_line_pattern.match(line):
            return False, f"Invalid format: '{line}'. Format must be 'Speaker: Text'"

    return True, ""

def split_dialogue(text: str) -> List[Tuple[str, str]]:
    """
    Splits the input text into a list of (speaker, line) tuples.
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    segments = []

    for line in lines:
        if ':' not in line:
            continue
        speaker, content = line.split(':', 1)
        segments.append((speaker.strip(), content.strip()))
    
    return segments

def generate_audio(text: str, lang_code: str = DEFAULT_LANG_CODE) -> str:
    # Clean previous files
    for file in os.listdir(audios_directory):
        os.remove(os.path.join(audios_directory, file))

    segments = split_dialogue(text)
    pipeline = KPipeline(lang_code=lang_code,repo_id="hexgrad/Kokoro-82M")

    speaker_voice_map = {}
    voice_index = 0
    chunks = []

    for speaker, line in segments:
        if speaker not in speaker_voice_map:
            speaker_voice_map[speaker] = VOICE_POOL[voice_index % len(VOICE_POOL)]
            voice_index += 1

        voice = speaker_voice_map[speaker]
        generator = pipeline(line, voice=voice)

        for _, _, audio in generator:
            chunks.append(audio)

    full_audio = np.concatenate(chunks, axis=0)
    file_path = os.path.join(audios_directory, "audio.wav")
    sf.write(file_path, full_audio, 24000)
    return file_path

# Streamlit UI
st.set_page_config(page_title="Multi-Speaker TTS", layout="centered")
st.title("🎙️TR Text-to-Speech Service")

st.markdown("""
Write your podcast-style conversation below, one speaker per line:
""")

user_input = st.text_area("Enter Dialogue", height=250)

# Validate input
is_valid, validation_error = validate_input_text(user_input)
if not is_valid and user_input.strip():
    st.error(validation_error)

if st.button("🎧 Generate Audio", disabled=not is_valid):
    with st.spinner("Synthesizing speech..."):
        audio_path = generate_audio(user_input)
    st.success("✅ Audio generated!")
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
        st.audio(audio_bytes, format='audio/wav')
        st.download_button("⬇️ Download Audio", data=audio_bytes, file_name="multi_speaker_audio.wav", mime="audio/wav")
