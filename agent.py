import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import requests

# Load environment
load_dotenv()

# Initialize Groq LLM

llm = ChatGroq(
    model="llama3-70b-8192",   # or "mixtral-8x7b-32768"
    temperature=0.7,
    max_tokens=1024,
)


# Same weather function from earlier
def get_weather(location: str) -> str:
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
    geo_data = requests.get(geo_url).json()
    if "results" not in geo_data: return f"No coords for {location}"

    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]

    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    weather_data = requests.get(weather_url).json()
    if "current_weather" not in weather_data: return f"No weather data for {location}"

    temp = weather_data["current_weather"]["temperature"]
    wind = weather_data["current_weather"]["windspeed"]
    return f"Weather in {location}: {temp}°C, wind {wind} km/h"

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a given city.",
    func=get_weather,
)

# Other tools
search_tool = DuckDuckGoSearchResults(name="web_search", description="Search the web.")
repl = PythonREPL()
repl_tool = Tool(name="python_repl", description="Run Python code.", func=repl.run)

tools = [search_tool, repl_tool, weather_tool]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI agent. Use tools when needed."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    response = agent_executor.invoke({"input": "What is the weather in Hyderabad?"})
    print(response["output"])
