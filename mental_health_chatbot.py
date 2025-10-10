'''
Code for creating a mental health chatbot using Python and Flask and streamlit.
'''

# from flask import Flask, request, jsonify
# import streamlit as st
from openai import OpenAI
import os
import dotenv
import logging  

# Load environment variables from .env file
dotenv.load_dotenv()

# call the OpenAI API key from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO)

# set up system message for the chatbot
system_message = "You are a compassionate and understanding mental health chatbot. Your purpose is to provide support, guidance, and resources to individuals seeking help with their mental health. You should respond with empathy, active listening, and encouragement. Avoid giving medical advice or diagnosing conditions. Instead, focus on providing emotional support and suggesting coping strategies."

# user message example
user_message = "Generate an uplifting story that would help a person with mental health issues."

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
)

# print the response from the chatbot
print(response.choices[0].message.content)

# TODO: Integrate with Flask and Streamlit for web interface
# TODO: have a main function to run the chatbot and take user input