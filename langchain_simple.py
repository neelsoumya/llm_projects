'''
Simple script to call LangChain
'''

import langchain
from langchain_openai import ChatOpenAI
import dotenv

# load environment variables from .env file
dotenv.load_dotenv()

# initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o", 
                 temperature=0, 
                 output_version="responses/v1")

# TODO: Add your LangChain code here
#.  https://python.langchain.com/docs/integrations/chat/openai/

# invoke the model with a prompt
response = llm.invoke("Hello, how are you?")
print("\nLLM response: \n")
print(response)

# add tool calls, memory, chains, agents, etc.
from pydantic import BaseModel, Field

class getWeather(BaseModel):
  """Get the weather in a given location
  """
  location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([getWeather])
str_message = llm_with_tools.invoke("What is the weather in Boston, MA?")

print("\n LLM with tool weather: \n")
print(str_message.content)

# calling web search tool
#  from langchain_tools import DuckDuckGoSearchResults
tool_search = {"type": "web_search_preview"}
llm_with_search = llm.bind_tools([tool_search])

str_message = llm_with_search.invoke("What is a positive uplifting news headline today?")
print("\n LLM with tool search: \n")
print(str_message.content)