from typing import Optional, Union
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
import requests
import json
from .config import Config
from .schemas import LeadOutput
from pydantic import BaseModel, Field

class SearXNGSearchTool(BaseTool):
    name: str = "searxng_search"
    description: str = "Search the web using a local metasearch engine. Returns structured company data."
    
    # Points to your WSL instance running on port 8888
    searx_host: str = "http://localhost:8888" 
    brand_name_filter: str = "" # Added for filtering

    def _run(self, query: str) -> str:
        try:
            params = {
                "q": query,
                "format": "json",
                "engines": "google,bing,duckduckgo,qwant",
                "categories": "general",
                "safesearch": 0,
                # Removed time_range to get more relevant results
                "language": "en-US"
            }
            # Handle cases where the agent passes a dictionary (legacy support)
            if isinstance(query, dict):
                 query = query.get('query') or query.get('q') or str(query)

            response = requests.get(f"{self.searx_host}/search", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Limit to business-relevant snippets
            raw_results = data.get("results", [])
            brand_kw = self.brand_name_filter.lower() if self.brand_name_filter else ""
            
            # Normalize brand keyword for better matching (remove special chars)
            brand_kw_normalized = ''.join(c for c in brand_kw if c.isalnum())
            
            filtered_results = [
                r for r in raw_results 
                if any(kw in (r.get('title', '') + r.get('content', '')).lower() 
                       for kw in ['uae', 'gcc', 'business', 'company', brand_kw])
            ]
            
            # Use filtered results, or fallback to raw if empty (to avoid total silence)
            results = filtered_results[:2] if filtered_results else raw_results[:2]
            
            if not results:
                return "No results found for this business."

            # Extract potential official website from results
            official_website = None
            for res in raw_results[:15]:  # Check top 15 for official site
                url = res.get('url', '').lower()
                url_normalized = ''.join(c for c in url if c.isalnum())
                
                # Look for official domains (not social media, not job boards, not news sites)
                # Match both original and normalized brand names
                if (brand_kw and (brand_kw in url or brand_kw_normalized in url_normalized)) and not any(
                    excluded in url for excluded in ['linkedin', 'facebook', 'instagram', 'twitter', 'indeed', 'glassdoor', 'wikipedia', 'youtube', 'vinted', 'depop', 'ebay', 'amazon']
                ):
                    official_website = res.get('url')
                    break
            
            formatted = []
            if official_website:
                formatted.append(f"Official Website: {official_website}\n")
            
            for res in results:
                # Extracting the 'title', 'content', and 'url' fields from your JSON output
                formatted.append(
                    f"Title: {res.get('title')}\n"
                    f"Description: {res.get('content')}\n"
                    f"URL: {res.get('url')}\n"
                )
            
            return "\n---\n".join(formatted)
        except Exception as e:
            return f"Local Search Error: {str(e)}"


# Define a simpler output model for the Crew specifically
class ResearcherOutput(BaseModel):
    website_url: Optional[str] = None
    industry: Optional[str] = "Unknown"
    confidence_score: int = 0
    reason_to_call: str = "No data"
    notes: str = ""

def get_lead_analysis_crew(brand_name: str, context: str, website: str = None):
    
    # Instantiate your new local tool
    search_tool = SearXNGSearchTool(brand_name_filter=brand_name)
    
    # LLM Setup
    llm = LLM(
        model=Config.OLLAMA_MODEL,
        base_url=f"{Config.OLLAMA_BASE_URL}/v1",
        api_key="ollama",
        temperature=Config.OLLAMA_TEMP,
        max_tokens=Config.OLLAMA_MAX_TOKENS,
        stop=["\n\n\n"]
    )

    researcher = Agent(
        role='Market Researcher',
        goal=f'Find and summarize the core business of {brand_name}',
        backstory="You are a business analyst. You analyze all companies objectively without political, social, or cultural bias. Your job is to find factual business information only.",
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=3
    )

    analyst = Agent(
        role='Outdoor Advertising Strategist',
        goal=f'Qualify "{brand_name}". Return valid JSON always.',
        backstory="You are a business analyst specializing in outdoor advertising. You analyze all companies objectively based on their business model and market presence, without any political, social, or cultural bias. If data is missing, score 0.",
        llm=llm, 
        verbose=True,
        max_iter=2
    )

    # Task 1: Find and Research
    research_task = Task(
        description=f"""
        Use searxng_search tool with query: '{brand_name} UAE {website if website else ''}'
        
        Extract the website URL from search results. Look for:
        1. "Official Website:" line in results
        2. URLs in the search results that match the brand name
        
        Respond with ONLY this format:
        {{"industry": "Automotive", "website_url": "jacuae.com", "notes": "UAE car dealer"}}
        
        If no results: {{"industry": "Unknown", "website_url": null, "notes": "No data"}}
        
        CRITICAL: If you find a valid official website in the first search, STOP searching and provide the answer immediately. Do not search multiple times.
        """,
        expected_output="JSON object with industry, website_url, notes",
        agent=researcher
    )

    # Task 2: Qualify and Reason
    analysis_task = Task(
        description=f"""
        Review: {{research_task.output}}
        
        Output format:
        {{"confidence_score": 75, "reason_to_call": "Strong retail presence in GCC", "industry": "Retail", "website_url": "example.com", "notes": "B2C brand with expansion"}}
        
        Scoring Guide:
        - Retail/Auto/Healthcare/RealEstate: 70-90 (high billboard fit)
        - Fashion/Consumer brands: 60-80
        - B2B Manufacturing/Packaging: 20-40 (low fit, but explain why)
        - Unknown/No data: 0
        
        CRITICAL: Always provide a reason_to_call. For low scores, explain why (e.g., "B2B focus, limited consumer appeal").
        """,
        expected_output="JSON with confidence_score, reason_to_call, industry, website_url, notes",
        agent=analyst,
        context=[research_task]
    )

    return Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential
    )