from pydantic.v1 import BaseModel, Field
from typing import List, Optional
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import os
from dotenv import load_dotenv


load_dotenv()
os.environ["TAVILY_API_KEY"] = str(os.getenv('TAVILY_API_KEY'))

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

