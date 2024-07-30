from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    verbose=True,
    google_api_key=GOOGLE_API_KEY,
)
search_engine = TavilySearchResults()
if __name__ == "__main__":
    print(llm.invoke("Hello"))
