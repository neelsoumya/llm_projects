'''
Simple script to call LangChain
'''

import langchain
from langchain_openai import ChatOpenAI
import dotenv

# load environment variables from .env file
dotenv.load_dotenv()

# initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# TODO: Add your LangChain code here
#.  https://python.langchain.com/docs/integrations/chat/openai/

# invoke the model with a prompt
response = llm.invoke("Hello, how are you?")
print(response)

# add tool calls, memory, chains, agents, etc.
from pydantic import BaseModel, Field

class getWeather(BaseModel):
  """Get the weather in a given location
  """
  location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([getWeather])
str_message = llm_with_tools.invoke("What is the weather in Boston, MA?")

print(str_message)