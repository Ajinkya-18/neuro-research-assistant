import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from tools.tools import tavily_search
    from agents.agent import get_agent_executor

except ImportError:
    st.error("Could not import 'get_agent_executor' and tavily_search from agent.py and tool.py")
    st.stop()

except Exception as e:
    st.error(f"Error importing from agent.py: {e}")
    st.stop()


load_dotenv()
os.environ['TAVILLY_API_KEY'] = str(os.getenv("TAVILLY_API_KEY"))
os.environ["GOOGLE_API_KEY"] = str(os.getenv("GOOGLE_API_KEY"))


@st.cache_resource
def load_agent():
    """Uses streamlit's cache to load the agent only once. It calls the imported function."""
    return get_agent_executor()


st.set_page_config(page_title="Agentic Search Bot", layout='centered')
st.title("Agentic Search Bot")
st.caption("A chatbot powered by Gemini and Tavily Search")

try:
    agent_executor = load_agent()

except Exception as e:
    st.error(f"Failed to load agent: {e}. Check API keys and permissions.")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    history_messages = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_messages.append(AIMessage(content=msg["content"]))

    with st.spinner("Agent is thinking..."):
        try:
            response = agent_executor.invoke({
                "user_input": user_prompt,
                "chat_history": history_messages
            })

            ai_response = response["output"]


            st.session_state.messages.append({"role": "assistant", "content": ai_response})

            with st.chat_message("assistant"):
                st.markdown(ai_response)

                steps = response.get("intermediate_steps", [])
                if steps:
                    with st.expander("Show Agent's work"):
                        for step in steps:
                            action, observation = step
                            st.markdown(f"**Tool:** '{action.tool}'")
                            st.markdown("**Tool Input:**")
                            st.json(action.tool_input)
                            st.markdown("**Tool Output:**")
                            st.code(observation, language="json")
            

        except Exception as e:
            st.error(f"An error occurred: {e}")

