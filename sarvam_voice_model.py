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

response = client.speech_to_text.transcribe(
    file = open("audio.wav", "rb"),
    model="saaras:v3",
    mode="transcribe" # or "translate", "verbatim", "translit", "codemix"
)

print("Response from SarvAM API: \n")
print(response)

# TODO: appify using streamlit or gradio

# TODO: apply for minorities grant to build voice assistant

# TODO: text to speech
# https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert
# https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/overview