import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.tools import tavily_search


def get_agent_executor():
    """Loads and returns a configured LangChain AgentExecutor."""
    print("--- Loading Agent ---")

    load_dotenv()
    os.environ["TAVILY_API_KEY"] = str(os.getenv('TAVILY_API_KEY'))
    os.environ["GOOGLE_API_KEY"] = str(os.getenv('GOOGLE_API_KEY'))

    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', 
                             temperature=0.5, 
                            #  convert_system_message_to_human=True
                            )
    
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

    print("--- Agent Loaded successfully ---")
    return agent_executor


def main():
    print("---Starting Agent---")

    agent_executor = get_agent_executor()

    question = """Explain the role of DMT as a neurohallucinogen and in altered states of consciousness research."""

    try:
        response = agent_executor.invoke({"user_input": question, 
                                          "chat_history": []
                                          })
        
        print("\n--- Final Answer ---")
        print(response["output"])


    except Exception as e:
        print(f"\n --- Error during agent execution ---")
        print(e)



if __name__ == "__main__":
    main()



