from dotenv import load_dotenv, find_dotenv

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper

load_dotenv(find_dotenv())

llm = OpenAI(model="text-davinci-003", temperature=0)

search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Retrieval QA System",
        func=search.run,
        description="Useful for answering questions."
    ),
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# run agent
response = agent.run("Can you give me the latest news on Israel and Gaza conflict?")
print(response)