'''
Sarvam voice model starter kit

Documentation:
- https://docs.sarvam.ai/api-reference-docs/getting-started/quickstart

Installation:

    python -m venv sarvam_venv

    source sarvam_venv/bin/activate

    pip install -r requirements_sarvam.txt

Usage:
    python sarvam_voice_model.py
    
Author: Soumya Banerjee

'''

from sarvamai import SarvamAI
import dotenv
import os
import logging
import wave
import sounddevice as sd
import numpy as np

# Load environment variables from .env file
# CAUTION: add SARVAM_API_KEY to .env file before running this code
# Also please add .env to .gitignore to avoid pushing your API key to github
dotenv.load_dotenv()
client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY")
)

# create wav file from mic 
def record_wav(filename="hello.wav", seconds=5, samplerate=16000):
    '''
    record audio from mic and save file
    '''
    print(f"Recording for {seconds} seconds... speak now")
    audio = sd.rec(
        int(seconds * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

    print(f"Saved {filename}")

record_wav(filename="audio.wav",
           seconds=5,
           samplerate=16000)    

str_mode = "translate" # or "transcribe", "verbatim", "translit", "codemix"

response = client.speech_to_text.transcribe(
    file = open("audio.wav", "rb"),
    model = "saaras:v3",
    mode = str_mode
    # mode="transcribe" # or "translate", "verbatim", "translit", "codemix"
)

print("Response from SarvAM API: \n")
print(response)

# TODO: appify using streamlit or gradio or huggingface spaces

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