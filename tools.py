from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

serpapi = SerpAPIWrapper()

@tool
def google_search_tool(query: str) -> list[dict[str, any]]:
    """
    A function that uses SerpAPI to perform a Google search and returns the top 5 results.
    
    Args:
        query (str): The search query.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the title, link, and snippet of the search result.
    """
    results = serpapi.search(
        engine="google",
        q=query,
        num=5,
        lang="en",
        country="US",
        api_key=os.getenv("SERPAPI_KEY"),
    )
    return results


@tool
def calculator(exp: str):
    """
    A simple calculator tool that evaluates mathematical expressions.
    
    Args:
        exp (str): The mathematical expression to evaluate.
        
    Returns:
        float: The result of the evaluation.
    """
    try:
        return eval(exp)
    except Exception as e:
        return str(e)