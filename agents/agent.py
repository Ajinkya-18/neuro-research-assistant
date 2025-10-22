import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
from typing import List, Optional

load_dotenv()
os.environ["TAVILY_API_KEY"] = str(os.getenv('TAVILY_API_KEY'))
os.environ["GOOGLE_API_KEY"] = str(os.getenv('GOOGLE_API_KEY'))

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', 
                             temperature=0.5, 
                            #  convert_system_message_to_human=True
                            )

_tavily_search_instance = TavilySearch(max_results=2)

class TavilyInput(BaseModel):
    query: str = Field(description="The search query.")
    include_domains: Optional[List[str]] = Field(
        default=None, description="A list of domains to specifically include in the search."
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None, 
        description="A list of domains to specifically exclude from the search."
    )
    
@tool(args_schema=TavilyInput)
def tavily_search(query: str, include_domains: Optional[List[str]]=None, exclude_domains: Optional[List[str]]=None) -> str:
    """A search engine optimized for comprehensive, accurate and trusted results."""
    input_dict = {"query": query}
    if include_domains:
        input_dict["include_domains"] = include_domains
    if exclude_domains:
        input_dict["exclude_domains"] = exclude_domains
    
    return _tavily_search_instance.invoke(input_dict)

tools = [tavily_search]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. You have access to a web search tool."), 
    ("user", "{user_input}"), 
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent = agent, 
    tools = tools, 
    verbose = True, 
    handle_parsing_errors=True, 
    max_iterations=3
)


def main():
    print("---Starting Agent---")

    question = """Explain the role of DMT as a neurohallucinogen and in altered states of consciousness research."""

    try:
        response = agent_executor.invoke({"user_input": question})

        print("\n--- Final Answer ---")
        print(response["output"])

    
    except Exception as e:
        print(f"\n --- Error during agent execution ---")
        print(e)


if __name__ == "__main__":
    main()

