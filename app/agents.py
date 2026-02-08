from typing import Optional, Union
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from .config import Config
from .schemas import LeadOutput
from pydantic import BaseModel, Field

# 1. Custom DuckDuckGo Search Tool
class DuckDuckGoSearchTool(BaseTool):
    name: str = "duck_duck_go_search"
    description: str = "Search the web to find company websites and industry info."
    
    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            
            if results:
                return str(results)
        except Exception as e:
            print(f"DDGS Error: {e}")
        
        # Fallback: Guess the URL
        # Clean the query to make a domain
        clean_name = "".join(c for c in query.lower() if c.isalnum())
        guessed_url = f"https://www.{clean_name}.com"
        
        try:
            # Quick check if it exists
            response = requests.head(guessed_url, timeout=3)
            # Accept 200-399 codes
            if response.status_code < 400:
                return f"Search API irrelevant/blocked. Found likely official website: {guessed_url}"
        except:
            pass
            
        return "No results found. Please try a different search query or use the provided website."

# 2. Custom Simple Scraper Tool
class SimpleScrapeTool(BaseTool):
    name: str = "website_scraper"
    description: str = "Reads website content. Input should be a single URL string."
    
    def _run(self, url: Union[str, dict]) -> str:
        # Sometimes CrewAI passes a dict like {'url': '...'} or internal context
        # This cleaning ensures we only get the string
        if isinstance(url, dict):
            url = url.get('url', str(url))
            
        try:
            # Normalize URL if needed (basic check)
            if not url.startswith('http'):
                url = f"https://{url}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            # Handle non-200 intelligently
            if response.status_code in [401, 403]:
                return "Access Denied by website. Proceed using only search context."
            elif response.status_code != 200:
                return f"Failed to retrieve content. Status code: {response.status_code}"
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Clean up the text: remove scripts/styles
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
                
            text = soup.get_text(separator=' ')
            cleaned_text = " ".join(text.split())
            
            # CRITICAL: Local LLMs have small memory. Limit to 1500 chars.
            return cleaned_text[:1500] 
            
        except Exception as e:
            return f"Scraping failed: {str(e)}"

# Define a simpler output model for the Crew specifically
class ResearcherOutput(BaseModel):
    website_url: Optional[str] = Field(None, description="The official company website URL.")
    industry: str = Field(..., description="The main industry the company operates in.")
    confidence_score: int = Field(..., description="0-100 score indicating how good a lead this is.")
    reason_to_call: str = Field(..., description="A 2-sentence reason to call.")
    notes: str = Field(..., description="Summary of findings.")

def get_lead_analysis_crew(brand_name: str, context: str, provided_website: str = None):
    
    # Instantiate tools
    search_tool = DuckDuckGoSearchTool()
    scrape_tool = SimpleScrapeTool()
    
    # LLM Setup - Force using OpenAI client format which Ollama supports
    local_llm = LLM(
        model=f"openai/{Config.OLLAMA_MODEL}", # Treat as generic OpenAI model
        base_url=f"{Config.OLLAMA_BASE_URL}/v1", 
        api_key="ollama", 
        timeout=300,
        config={"num_ctx": 4096} # Gives the AI a larger "working memory"
    )

    researcher = Agent(
        role='Market Researcher',
        goal=f'Find the company website for "{brand_name}" and analyze its business.',
        backstory="Expert in finding B2B company details and analyzing their business model.",
        tools=[search_tool, scrape_tool],
        llm=local_llm,
        verbose=True
    )

    analyst = Agent(
        role='Sales Strategist',
        goal=f'Qualify the lead "{brand_name}" and write a sales hook.',
        backstory="You generate high-quality sales hooks. You determine if a lead is worth calling.",
        llm=local_llm, 
        verbose=True
    )

    # Task 1: Find and Research
    # We explicitly ask to verify the website or find it.
    research_task = Task(
        description=f"""
        1. PROVIDED_WEBSITE: '{provided_website}'
        2. IF PROVIDED_WEBSITE is valid (not None):
           - USE 'website_scraper' on it IMMEDIATELY.
           - DO NOT use 'duck_duck_go_search'.
        3. IF PROVIDED_WEBSITE is None:
           - USE 'duck_duck_go_search' to find the URL.
           - Then USE 'website_scraper' on the found URL.
        4. ANALYZE content to extract Industry and compare with context: "{context}".
        5. VERY IMPORTANT: Do NOT return the full text of the website. 
           Summarize the services in under 100 words.
        """,
        expected_output="A bullet-point summary of services (max 100 words) and the URL.",
        agent=researcher
    )

    # Task 2: Qualify and Reason
    analysis_task = Task(
        description=f"""
        Based on the research:
        1. Determine a 'Confidence Score' (0-100) for this lead.
           - High score if they clearly fit the context: "{context}".
           - Low score if irrelevant.
        2. Write a 'Reason to Call' (max 2 sentences). Mention specific news/service.
        3. Summarize findings in 'Notes'.
        
        Return the result in a structured format.
        """,
        expected_output="Structured JSON with confidence_score, reason_to_call, industry, notes, website_url.",
        agent=analyst,
        context=[research_task],
        output_pydantic=ResearcherOutput # Enforce structure
    )

    return Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential
    )