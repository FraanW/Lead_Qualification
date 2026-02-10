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
                       for kw in ['uae', 'gcc', 'business', 'company', 'contact', 'phone', 'email', 'maps', brand_kw])
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

class WebCrawlTool(BaseTool):
    name: str = "web_crawl"
    description: str = "Crawl a website to extract text content, specifically targeting header and footer for contact info."

    def _run(self, url: str) -> str:
        try:
            import re
            response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style, and navigation noise if possible
            for tag in soup(["script", "style", "svg", "path"]):
                tag.decompose()
            
            # Try to find header and footer tags
            header = soup.find('header')
            footer = soup.find('footer')
            
            header_text = header.get_text(separator=' ').strip() if header else ""
            footer_text = footer.get_text(separator=' ').strip() if footer else ""
            
            # Fallback to common class names if semantic tags are missing
            if not header_text:
                header_alt = soup.find(['div', 'nav'], class_=re.compile(r'header|top|nav', re.I))
                if header_alt: header_text = header_alt.get_text(separator=' ').strip()
                
            if not footer_text:
                footer_alt = soup.find('div', class_=re.compile(r'footer|bottom', re.I))
                if footer_alt: footer_text = footer_alt.get_text(separator=' ').strip()

            # Get general text for context
            full_text = soup.get_text(separator=' ')
            lines = [line.strip() for line in full_text.splitlines() if line.strip()]
            main_text = ' '.join(lines)[:2000]
            
            # Format output for LLM to see clear sections
            result = f"DOMAIN: {url}\n\n"
            result += f"--- WEBSITE HEADER ---\n{header_text[:1000] if header_text else 'No clear header found.'}\n\n"
            result += f"--- WEBSITE FOOTER ---\n{footer_text[:1000] if footer_text else 'No clear footer found.'}\n\n"
            result += f"--- PAGE PREVIEW ---\n{main_text}"
            
            return result
        except Exception as e:
            return f"Crawl Error: {str(e)}"


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

def get_social_lead_analysis_crew(brand_name: str, influencer: str, post_reason: str, website: str = None):
    # Instantiate tools
    search_tool = SearXNGSearchTool(brand_name_filter=brand_name)
    crawl_tool = WebCrawlTool()
    
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
        role='Business Intelligence Researcher',
        goal=f'Find and verify the official website of {brand_name}.',
        backstory="You are experts at identifying the ONE official website for a brand, avoiding social media pages or directories.",
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=3
    )

    contact_extractor = Agent(
        role='Contact Information Specialist',
        goal=f'Extract phone numbers and emails from the official website headers and footers of {brand_name}.',
        backstory="You are a specialist in finding contact details. You focus specifically on the header and footer of websites, as that is where contact info usually lives.",
        tools=[crawl_tool],
        llm=llm,
        verbose=True,
        max_iter=3
    )

    brand_strategist = Agent(
        role='Brand Strategist',
        goal=f'Understand {brand_name}\'s business model and craft a compelling reason to call them.',
        backstory="You are a marketing expert. You analyze a company's website content to understand what they do, their target audience, and why they would benefit from outdoor advertising.",
        tools=[crawl_tool],
        llm=llm,
        verbose=True,
        max_iter=3
    )

    analyst = Agent(
        role='Social Lead Validator',
        goal=f'Determine if {brand_name} is a high-quality lead based on all research. Qualify and score lead.',
        backstory="You synthesize all data. You look at the business relevance and the ease of contact to provide a final score.",
        llm=llm, 
        verbose=True,
        max_iter=2
    )

    # Task 1: Find Official Website
    research_task = Task(
        description=f"""
        1. Search for {brand_name} (UAE focus) using searxng_search.
        2. Identify the ONE official website URL (e.g., brandname.com or linktr.ee/brand).
        
        Respond with valid JSON:
        {{
            "brand_name": "{brand_name}",
            "official_website": "...", 
            "industry_guess": "..."
        }}
        """,
        expected_output="JSON with official website URL",
        agent=researcher
    )

    # Task 2: Targeted Contact Extraction
    contact_task = Task(
        description=f"""
        Review the official website found: {{research_task.output}}
        
        1. Use web_crawl on the official website.
        2. Analyze the HEADER and FOOTER sections provided by the tool.
        3. Extract any Phone Numbers and Emails. 
        4. If the website is a Linktree, extract all listed links and contact options.
        
        Respond with valid JSON:
        {{
            "phone": "...", 
            "email": "...",
            "other_contacts": "..."
        }}
        """,
        expected_output="JSON with extracted contact details",
        agent=contact_extractor,
        context=[research_task]
    )

    strategy_task = Task(
        description=f"""
        Review the website details for {brand_name}: {{contact_task.output}}
        
        1. Use web_crawl on the official website if needed to understand the business offerings.
        2. Craft a ONE SENTENCE 'reason to call' that explains why we should reach out to them.
        3. Identify their main industry.
        
        Respond with valid JSON:
        {{
            "ai_reason_to_call": "...",
            "industry": "..."
        }}
        """,
        expected_output="JSON with AI reason to call and industry",
        agent=brand_strategist,
        context=[research_task, contact_task]
    )

    # Task 3: Final Validation and Scoring
    validation_task = Task(
        description=f"""
        Influencer '{influencer}' promoted '{brand_name}' for: '{post_reason}'.
        Research: {{research_task.output}}
        Contacts: {{contact_task.output}}
        
        Final Lead Qualification:
        - confidence_score (0-100)
        - contactibility_score (0-100): High if phone AND email found in website header/footer.
        
        Ensure the 'company' object is fully populated.
        
        Output JSON:
        {{
            "brand_name": "{brand_name}",
            "confidence_score": ...,
            "contactibility_score": ...,
            "category_main_industry": "...",
            "ai_reason_to_call": "...",
            "notes": "Extracted contacts from website header/footer.",
            "company": {{
                "phone": "...",
                "email": "...",
                "website": "...",
                "Other": "..."
            }}
        }}
        """,
        expected_output="Final Lead JSON",
        agent=analyst,
        context=[research_task, contact_task]
    )

    return Crew(
        agents=[researcher, analyst],
        tasks=[research_task, validation_task],
        process=Process.sequential
    )