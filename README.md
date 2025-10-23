# neuro-research-assistant

A small agentic research assistant project that wires a LangChain-based agent to a web-search tool and a Streamlit UI.

This repository contains the minimal components used to construct and run an agent backed by Gemini (via the `langchain-google-genai` integration) and a Tavily search tool (via `langchain-tavily`). The project includes a Streamlit app to interact with the agent.

Only functionality that appears in the repository is described below.

## What’s in this repository

- `agents/agent.py` — Builds and returns a configured LangChain `AgentExecutor` using:
  - `ChatGoogleGenerativeAI` (model set to `gemini-2.0-flash`)
  - A prompt template with a system message and user input placeholder
  - The `tavily_search` tool (imported from `tools.tools`)
  - The agent executor is created with `max_iterations=3` and verbose logging enabled.

- `tools/tools.py` — Defines a `tavily_search` tool using `langchain-tavily`:
  - A small Pydantic model `TavilyInput` describes the tool's inputs (`query`, optional `include_domains`, `exclude_domains`).
  - A `TavilySearch` instance is created (`max_results=2`) and exposed as a LangChain tool using the `@tool` decorator.

- `app/app.py` — A Streamlit front-end that:
  - Loads the agent via `agents.agent.get_agent_executor` (cached with `st.cache_resource`).
  - Provides a chat UI (`st.chat_input`, `st.chat_message`) to send user prompts to the agent and display responses.
  - Shows intermediate tool steps (tool name, inputs, outputs) in an expander when available.

- `requirements.txt` — Lists runtime dependencies observed in the repository:
  - `streamlit`
  - `langchain`
  - `langchain-google-genai`
  - `langchain-tavily`
  - `python-dotenv`
  - `pydantic<2`

- `LICENSE` — MIT License (copyright 2025 Ajinkya A Tamhankar).

## Environment and configuration

- The code expects API keys to be provided via environment variables or a `.env` file and uses `python-dotenv` to load them. The project references the following environment variables in multiple files:
  - `TAVILY_API_KEY` (or `TAVILLY_API_KEY` in a couple of places in the code — check for the exact spelling in your environment)
  - `GOOGLE_API_KEY`
  
  Note: The repository sets these environment variables from `os.getenv(...)` in the code before creating client instances. Ensure the appropriate API keys are available in your environment.

## How to run (local)

1. Create and activate a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Create a `.env` file (or set environment variables) with the expected API keys.
4. Run the Streamlit app from the `app` directory.

Example (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# create a .env file with TAVILY_API_KEY and GOOGLE_API_KEY (or set environment variables)
streamlit run app\app.py
```

## Notes and caveats

- The README only documents code and files present in the repository. It does not claim any hosted service or external infrastructure.
- There are a couple of inconsistent environment variable spellings in the code (`TAVILY_API_KEY` vs `TAVILLY_API_KEY`) — verify and set both if necessary or edit the code to standardize the name you prefer.
- The Streamlit UI attempts to import and load the agent and will stop with an error message if imports fail or API keys are missing.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
