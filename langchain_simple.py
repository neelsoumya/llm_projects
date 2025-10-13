'''
Simple script to call LangChain
'''

import langchain
from langchain_openai import ChatOpenAI
import dotenv

# load environment variables from .env file
dotenv.load_dotenv()

# initialize the ChatOpenAI model
llm = ChatOpenAI()

# llm_response = llm.g

#print(llm_response)

