"""
Sarvam voice model Streamlit app.

Documentation:
- https://docs.sarvam.ai/api-reference-docs/getting-started/quickstart
- https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/overview

Installation:
    python -m venv sarvam_venv
    source sarvam_venv/bin/activate
    pip install -r requirements_sarvam.txt

Usage:
    streamlit run sarvam_voice_model.py
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import dotenv
import streamlit as st
from sarvamai import SarvamAI

# Load environment variables from .env file
dotenv.load_dotenv()

SUPPORTED_MODES = ["translate", "transcribe", "verbatim", "translit", "codemix"]
SUPPORTED_TYPES = ["wav", "mp3", "m4a", "ogg", "flac", "webm"]


@st.cache_resource
def get_client() -> SarvamAI:
    """Create and cache Sarvam client."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError(
            "SARVAM_API_KEY is missing. Add it to your environment or .env file."
        )
    return SarvamAI(api_subscription_key=api_key)


def extract_text(response: Any) -> str:
    """Best-effort extraction of text from SDK response."""
    if response is None:
        return ""
    if isinstance(response, dict):
        for key in ["text", "transcript", "translation", "output", "result"]:
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    for key in ["text", "transcript", "translation", "output", "result"]:
        value = getattr(response, key, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def main() -> None:
    st.set_page_config(page_title="Sarvam Voice App", page_icon="🎙️", layout="centered")
    st.title("🎙️ Sarvam Voice Model UI")
    st.caption("Speak or upload audio and run Saaras v3 transcription/translation.")

    api_key_set = bool(os.getenv("SARVAM_API_KEY"))
    st.sidebar.header("Configuration")
    st.sidebar.write(f"SARVAM_API_KEY detected: {'Yes' if api_key_set else 'No'}")
    model = st.sidebar.text_input("Model", value="saaras:v3")
    mode = st.sidebar.selectbox("Mode", SUPPORTED_MODES, index=0)
    language_code = st.sidebar.text_input(
        "Language code (optional)", placeholder="e.g. hi-IN"
    )

    st.subheader("1) Provide Audio")
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=SUPPORTED_TYPES,
        help="Supported formats: wav, mp3, m4a, ogg, flac, webm",
    )
    recorded_audio = st.audio_input("Or record audio in browser")

    audio_source = recorded_audio if recorded_audio is not None else uploaded_file
    if audio_source is not None:
        st.audio(audio_source)

    st.subheader("2) Run Speech-to-Text")
    if st.button("Transcribe / Translate", type="primary", use_container_width=True):
        if audio_source is None:
            st.error("Please upload or record audio first.")
            return

        try:
            client = get_client()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        file_ext = os.path.splitext(audio_source.name)[1] if audio_source.name else ""
        suffix = file_ext if file_ext else ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_source.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner("Calling Sarvam API..."):
                with open(tmp_path, "rb") as audio_file:
                    request_payload = {
                        "file": audio_file,
                        "model": model,
                        "mode": mode,
                    }
                    if language_code.strip():
                        request_payload["language_code"] = language_code.strip()
                    response = client.speech_to_text.transcribe(**request_payload)

            st.success("Done.")
            st.subheader("Result")
            text = extract_text(response)
            if text:
                st.text_area("Transcript / Translation", value=text, height=220)
                st.download_button(
                    "Download result as .txt",
                    data=text,
                    file_name="sarvam_output.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            else:
                st.info("No plain text field found. Showing full response object.")
            st.json(response)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Sarvam API call failed: {exc}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    main()

# TODO: apply for minorities grant to build voice assistant

# TODO: agents
# https://dashboard.sarvam.ai/agents

# TODO: chat
# https://dashboard.sarvam.ai/chat
# https://indus.sarvam.ai/


# 1. Clinic intake for low-English speakers. 
# A patient speaks in Hindi, Tamil, Bengali, etc.; 
# Saaras v3 transcribes or translates the audio, 
# Sarvam-Translate/Mayura turns it into English for the clinician, 
# and Bulbul v3 reads back instructions in the patient’s language. 
# This is especially useful for triage, medication instructions, 
# appointment reminders, and discharge summaries. 
# Saaras v3 supports flexible output modes like transcribe, 
# translate, verbatim, translit, and codemix, while Bulbul v3 
# provides natural TTS in 11 languages.

# 2. Voice assistant for visually impaired.
# A user speaks in their native language,
# Saaras v3 transcribes or translates the audio,
# Sarvam-Translate/Mayura turns it into English for the assistant,
# and Bulbul v3 reads back the assistant's response in the user's language.

# 3. Multilingual voice-based note-taking app.
# A user speaks in their native language,
# Saaras v3 transcribes or translates the audio,
# Sarvam-Translate/Mayura turns it into English for summarization,
# and Bulbul v3 reads back the summary in the user's language.

# 4. Multilingual voice-based language learning app.

# 5. Community helpline / NGO call center. A caller can speak freely in a local language or code-mixed speech; Saaras v3 can turn that into text, Sarvam-105B can summarize the issue and suggest a category, and Bulbul v3 can respond with a spoken answer in the caller’s language. This is good for domestic abuse support lines, benefits advice, disability services, or migrant support. The reason Sarvam is a good fit is that its speech models are designed for Indian-language and mixed-language use, and the chat model is optimized for long-horizon reasoning and tool use.

# 6. Government or legal aid document assistant. Sarvam Vision can extract text from forms, notices, and IDs in 23 languages, then Sarvam Translate can normalize it into English for staff review, and Bulbul v3 can explain the form back to the user orally. This is useful for housing, immigration support, benefits, and local council services. Sarvam Vision is specifically positioned for document intelligence across 23 languages.

# 7. Accessibility tool for older adults or low-literacy users. A voice-first assistant can take spoken questions, transcribe them with Saaras v3, answer in simple language with Sarvam-30B or 105B, and speak the answer back with Bulbul v3. Sarvam’s official docs describe the stack as built for India’s linguistic diversity, accents, and real-world usage patterns 

# A simple proof-of-concept idea would be this:

# “Speak → Transcribe → Translate → Summarize → Speak back”

# For example:

# user speaks in Hindi or Tamil
# Saaras v3 transcribes the audio
# Mayura or Sarvam Translate translates it to English for staff
# Sarvam-105B produces a short structured summary
# Bulbul v3 reads the summary back in the user’s language.

# TODO: text to speech
# https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert
# https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/overview