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
            potential_sites = []
            for res in raw_results[:15]:  # Check top 15 for official site
                url = res.get('url', '').lower()
                url_normalized = ''.join(c for c in url if c.isalnum())
                
                # Look for official domains (not social media, not job boards, not news sites)
                if (brand_kw and (brand_kw in url or brand_kw_normalized in url_normalized)) and not any(
                    excluded in url for excluded in ['linkedin', 'facebook', 'instagram', 'twitter', 'indeed', 'glassdoor', 'wikipedia', 'youtube', 'vinted', 'depop', 'ebay', 'amazon', 'pinterest']
                ):
                    potential_sites.append(res.get('url'))
            
            # Prioritize UAE-specific domains/paths
            official_website = None
            if potential_sites:
                # 1st priority: .ae domains or /uae/ /en-ae/ paths
                for site in potential_sites:
                    site_lower = site.lower()
                    if '.ae' in site_lower or '/uae' in site_lower or '/en-ae' in site_lower or 'dubai' in site_lower:
                        official_website = site
                        break
                # 2nd priority: the very first valid match (likely the .com)
                if not official_website:
                    official_website = potential_sites[0]
            
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
    description: str = "Crawl a website to extract text content, targeting header/footer, 'Contact Us' pages, and main business info."

    def _run(self, url: Union[str, dict] = None, **kwargs) -> str:
        try:
            import re
            from urllib.parse import urljoin
            
            # Handle cases where the agent passes a dictionary or unexpected keyword arguments
            if isinstance(url, dict):
                url = url.get('url') or url.get('target_url') or url.get('website') or str(url)
            elif url is None and kwargs:
                url = kwargs.get('url') or kwargs.get('target_url') or kwargs.get('website') or next(iter(kwargs.values()))
            
            if not url or not isinstance(url, str):
                return "Error: No valid URL provided to crawl."

            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 1. Discover "Contact Us" or "About" links
            contact_url = None
            for link in soup.find_all('a', href=True):
                text = link.get_text().lower()
                href = link['href'].lower()
                if re.search(r'contact|about-us|find-us|locations', text) or re.search(r'contact|about|location', href):
                    contact_url = urljoin(url, link['href'])
                    break
            
            # Clean up noise from homepage
            for tag in soup(["script", "style", "svg", "path", "iframe"]):
                tag.decompose()
            
            header = soup.find(['header', 'div'], id=re.compile(r'header', re.I), class_=re.compile(r'header', re.I))
            if not header: header = soup.find('header')
            
            footer = soup.find(['footer', 'div'], id=re.compile(r'footer', re.I), class_=re.compile(r'footer', re.I))
            if not footer: footer = soup.find('footer')
            
            header_text = header.get_text(separator=' ').strip() if header else ""
            footer_text = footer.get_text(separator=' ').strip() if footer else ""
            
            if not header_text:
                header_alt = soup.find(['div', 'nav'], class_=re.compile(r'header|top|nav', re.I))
                if header_alt: header_text = header_alt.get_text(separator=' ').strip()
                
            if not footer_text:
                footer_alt = soup.find(['div', 'section'], class_=re.compile(r'footer|bottom', re.I))
                if footer_alt: footer_text = footer_alt.get_text(separator=' ').strip()

            full_text = soup.get_text(separator=' ')
            lines = [line.strip() for line in full_text.splitlines() if line.strip()]
            main_text = ' '.join(lines)[:2500] # Increased for address context
            
            result = f"DOMAIN: {url}\n\n"
            result += f"--- WEBSITE HEADER (Potential Contacts/Links) ---\n{header_text[:1000] if header_text else 'No header.'}\n\n"
            result += f"--- WEBSITE FOOTER (Potential Addresses/Contacts) ---\n{footer_text[:1000] if footer_text else 'No footer.'}\n\n"
            result += f"--- MAIN PAGE CONTENT (Business Focus & Addresses) ---\n{main_text}\n\n"
            
            # 2. Crawl Contact Page if found
            if contact_url and contact_url != url:
                try:
                    c_res = requests.get(contact_url, timeout=8, headers=headers)
                    if c_res.status_code == 200:
                        c_soup = BeautifulSoup(c_res.text, 'html.parser')
                        for tag in c_soup(["script", "style", "svg", "path", "iframe"]):
                            tag.decompose()
                        c_text = c_soup.get_text(separator=' ')
                        c_lines = [line.strip() for line in c_text.splitlines() if line.strip()]
                        result += f"--- DEDICATED CONTACT PAGE ({contact_url}) ---\n"
                        result += ' '.join(c_lines)[:2000]
                except:
                    result += f"\n(Note: Failed to crawl found contact page: {contact_url})"
            
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
        role='Market Researcher',
        goal=f'Find and summarize the core business of {brand_name}',
        backstory="You are a business analyst. You analyze all companies objectively without political, social, or cultural bias. Your job is to find factual business information only.",
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=3
    )

    contact_extractor = Agent(
        role='Contact Information Specialist',
        goal=f'Extract phone numbers, emails, and business addresses from the official website content of {brand_name}.',
        backstory="You are a specialist in finding contact details. You focus on the header, footer, and main content of pages, as well as dedicated contact pages to find phone, email, and physical addresses.",
        tools=[crawl_tool],
        llm=llm,
        verbose=True,
        max_iter=3
    )

    analyst = Agent(
        role='Outdoor Advertising Strategist',
        goal=f'Qualify "{brand_name}". Return valid JSON always.',
        backstory="You are a business analyst specializing in outdoor advertising. You analyze all companies objectively based on their business model and market presence. If an address is found, you include it in the notes.",
        llm=llm, 
        verbose=True,
        max_iter=2
    )

    research_task = Task(
        description=f"""
        Use searxng_search tool with query: '{brand_name} UAE {website if website else ''}'
        
        1. Extract the website URL from search results.
        2. CRITICAL: Look for local UAE contact details in search descriptions (e.g., +971 phone numbers, Dubai Mall addresses).
        3. CRITICAL FALLBACK: If the main website found is a global domain (e.g. .com) and lacks +971 contacts in the snippet, perform a secondary search for '{brand_name} .ae' or '{brand_name} UAE official website' to find the local version.
        
        Respond with ONLY this format:
        {{
            "industry": "...", 
            "website_url": "...", 
            "local_contact_snippet": "Summarize any +971 or UAE info found here",
            "notes": "..."
        }}
        
        If no results: {{"industry": "Unknown", "website_url": null, "local_contact_snippet": "", "notes": "No data"}}
        """,
        expected_output="JSON object with industry, website_url, local_contact_snippet, notes",
        agent=researcher
    )

    # Task 2: Extract Contact Information
    contact_task = Task(
        description=f"""
        Research Output: {{research_task.output}}
        
        1. Extract the website_url.
        2. Use web_crawl on the URL.
        3. CRITICAL: Prioritize UAE/GCC contact details (+971 numbers and UAE addresses). 
        4. If the researcher found a local snippet ({{research_task.output.local_contact_snippet}}), incorporate that info.
        
        Respond with valid JSON:
        {{
            "phone": "...", 
            "email": "...",
            "address": "...",
            "other_contacts": "..."
        }}
        
        If no website or crawl fails, use the info from the researcher snippet if available.
        """,
        expected_output="JSON with extracted contact details (UAE prioritized)",
        agent=contact_extractor,
        context=[research_task]
    )

    analysis_task = Task(
        description=f"""
        Review Research: {{research_task.output}}
        Review Contacts: {{contact_task.output}}
        
        Final Qualification:
        1. Prioritize UAE Presence: If both a global and a UAE contact (+971) were found, you MUST use the UAE one.
        2. Ensure the 'company' object uses the best UAE-specific contact info.
        3. Address formatting: Only prefix address in 'notes' if it exists.
        
        Scoring Guide:
        - Retail/Auto/Healthcare/RealEstate: 70-90 (high billboard fit)
        - Fashion/Consumer brands: 60-80
        - B2B Manufacturing/Packaging: 20-40 (low fit, but explain why)
        - Unknown/No data: 0
        
        CRITICAL: confidence_score MUST be an INTEGER between 0 and 100.
        
        Output format:
        {{
            "confidence_score": 80,
            "reason_to_call": "...",
            "category_main_industry": "...",
            "notes": "...",
            "company": {{
                "phone": "...",
                "email": "...",
                "website": "...",
                "Other": "..."
            }}
        }}
        """,
        expected_output="Final Qualified Lead JSON",
        agent=analyst,
        context=[research_task, contact_task]
    )

    return Crew(
        agents=[researcher, contact_extractor, analyst],
        tasks=[research_task, contact_task, analysis_task],
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
        goal=f'Extract phone numbers, emails, and address from official website and contact pages of {brand_name}.',
        backstory="You are a specialist in finding contact details. You look at the header, footer, main content, and contact pages.",
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
        backstory="You synthesize all data. You look at the business relevance and the ease of contact to provide a final score. You ensure address info is in the notes.",
        llm=llm, 
        verbose=True,
        max_iter=2
    )

    # Task 1: Find Official Website
    research_task = Task(
        description=f"""
        1. Search for {brand_name} (UAE focus) using searxng_search.
        2. Identify the ONE official website URL.
        3. CRITICAL: Look for local UAE contact details in search descriptions (e.g., +971 phone numbers, UAE branch addresses).
        4. CRITICAL FALLBACK: If the main website found is global (e.g. .com), secondary search for '{brand_name} .ae' or '{brand_name} UAE'.
        
        Respond with valid JSON:
        {{
            "brand_name": "{brand_name}",
            "official_website": "...", 
            "local_contact_snippet": "Summarize any +971 or UAE info found",
            "industry_guess": "..."
        }}
        """,
        expected_output="JSON with official website URL and local contact snippet",
        agent=researcher
    )

    # Task 2: Targeted Contact Extraction
    contact_task = Task(
        description=f"""
        Review Research: {{research_task.output}}
        
        1. Use web_crawl on the official website.
        2. CRITICAL: Prioritize UAE/GCC contact details (+971 numbers and UAE addresses). 
        3. If the researcher found a local snippet ({{research_task.output.local_contact_snippet}}), incorporate that info.
        
        Respond with valid JSON:
        {{
            "phone": "...", 
            "email": "...",
            "address": "...",
            "other_contacts": "..."
        }}
        """,
        expected_output="JSON with extracted contact details (UAE prioritized)",
        agent=contact_extractor,
        context=[research_task]
    )

    # Task 3: Brand Strategy (no change needed in logic, but context remains)
    strategy_task = Task(
        description=f"""
        Review Research: {{research_task.output}}
        Review Contacts: {{contact_task.output}}
        
        1. Use web_crawl on the official website if needed.
        2. Craft a ONE SENTENCE 'reason to call' for {brand_name}.
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

    # Task 4: Final Validation and Scoring
    validation_task = Task(
        description=f"""
        Influencer '{influencer}' promoted '{brand_name}' for: '{post_reason}'.
        Research: {{research_task.output}}
        Contacts: {{contact_task.output}}
        Strategy: {{strategy_task.output}}
        
        Final Lead Qualification:
        1. Prioritize UAE Presence: If both a global and a UAE contact (+971) were found, you MUST use the UAE one.
        2. Ensure the 'company' object uses the best UAE-specific contact info.
        3. Address formatting: Only prefix address in 'notes' if it exists.
        
        Scoring Guide:
        - Retail/Auto/Healthcare/RealEstate: 70-90 (high billboard fit)
        - Fashion/Consumer brands: 60-80
        - B2B Manufacturing/Packaging: 20-40 (low fit, but explain why)
        - Unknown/No data: 0
        
        CRITICAL: confidence_score MUST be an INTEGER between 0 and 100.
        
        Output JSON:
        {{
            "brand_name": "{brand_name}",
            "confidence_score": 85,
            "contactibility_score": ...,
            "category_main_industry": "{{strategy_task.output.industry}}",
            "ai_reason_to_call": "{{strategy_task.output.ai_reason_to_call}}",
            "notes": "Extracted contacts from website content.",
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
        context=[research_task, contact_task, strategy_task]
    )

    return Crew(
        agents=[researcher, contact_extractor, brand_strategist, analyst],
        tasks=[research_task, contact_task, strategy_task, validation_task],
        process=Process.sequential
    )


def get_business_lead_analysis_crew(brand_name: str, website: str = None):
    """
    Business lead analysis crew - similar to social leads but focused on business data
    """
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
        goal=f'Find and verify the official website of {brand_name}. Focus on UAE presence.',
        backstory="You are experts at identifying the ONE official website for a brand, avoiding social media pages or directories. You prioritize UAE-specific domains (.ae) if they exist.",
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=3
    )

    contact_extractor = Agent(
        role='Contact Information Specialist',
        goal=f'Extract phone numbers, emails, and address from official website and contact pages of {brand_name}.',
        backstory="You are a specialist in finding contact details. You focus on the header, footer, main content, and contact pages.",
        tools=[crawl_tool],
        llm=llm,
        verbose=True,
        max_iter=3
    )

    brand_strategist = Agent(
        role='Brand Strategist',
        goal=f'Understand {brand_name}\'s business model and craft a compelling reason to call them.',
        backstory="You are a marketing expert. You analyze a company's website content to understand what they do and why they would benefit from outdoor advertising in the UAE.",
        tools=[crawl_tool],
        llm=llm,
        verbose=True,
        max_iter=3
    )

    analyst = Agent(
        role='Business Lead Validator',
        goal=f'Determine if {brand_name} is a high-quality lead. Qualify and score lead.',
        backstory=f"You synthesize all data. You filter out public spaces and malls. "
                  f"CRITICAL: If {brand_name} is identified as a public mall (e.g., Deira City Centre, Dubai Mall), "
                  f"or a public space (e.g., Waterfront Beach, Corniche, Parks, Markets), you MUST score confidence_score as 0-10 "
                  f"and set notes to 'Public space/Mall - not suitable for outreach.'",
        llm=llm, 
        verbose=True,
        max_iter=2
    )

    # Task 1: Find Official Website with .ae Fallback
    research_task = Task(
        description=f"""
        1. Search for {brand_name} (UAE focus) using searxng_search.
        2. Identify the ONE official website URL.
        3. CRITICAL: Look for local UAE contact details in search descriptions (e.g., +971 phone numbers, UAE branch addresses).
        4. CRITICAL FALLBACK: If the main website found is a global domain (e.g. .com) and lacks +971 contacts, perform a secondary search for '{brand_name} .ae' or '{brand_name} UAE official website'.
        
        Respond with valid JSON:
        {{
            "brand_name": "{brand_name}",
            "official_website": "...", 
            "local_contact_snippet": "Summarize any +971 or UAE info found",
            "industry_guess": "..."
        }}
        """,
        expected_output="JSON with official website URL and local contact snippet",
        agent=researcher
    )

    # Task 2: Targeted Contact Extraction
    contact_task = Task(
        description=f"""
        Review Research: {{research_task.output}}
        
        1. Use web_crawl on the official website.
        2. CRITICAL: Prioritize UAE/GCC contact details (+971 numbers and UAE addresses). 
        3. If the researcher found a local snippet ({{research_task.output.local_contact_snippet}}), incorporate that info.
        
        Respond with valid JSON:
        {{
            "phone": "...", 
            "email": "...",
            "address": "...",
            "other_contacts": "..."
        }}
        """,
        expected_output="JSON with extracted contact details (UAE prioritized)",
        agent=contact_extractor,
        context=[research_task]
    )

    # Task 3: Brand Strategy
    strategy_task = Task(
        description=f"""
        Review Research: {{research_task.output}}
        Review Contacts: {{contact_task.output}}
        
        1. Use web_crawl on the official website if needed.
        2. Craft a ONE SENTENCE 'reason to call' for {brand_name}.
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

    # Task 4: Final Validation and Scoring
    validation_task = Task(
        description=f"""
        Research: {{research_task.output}}
        Contacts: {{contact_task.output}}
        Strategy: {{strategy_task.output}}
        
        Final Lead Qualification:
        1. Prioritize UAE Presence: If both a global and a UAE contact (+971) were found, you MUST use the UAE one.
        2. Ensure the 'company' object uses the best UAE-specific contact info.
        3. Address formatting: Only prefix address in 'notes' if it exists.
        4. IMPORTANT: Penalize public spaces/malls as per backstory (0-10 score).
        
        Scoring Guide:
        - Retail/Auto/Healthcare/RealEstate: 70-90 (high billboard fit)
        - Fashion/Consumer brands: 60-80
        - B2B Manufacturing/Packaging: 20-40 (low fit, but explain why)
        - Unknown/No data: 0
        
        CRITICAL: confidence_score MUST be an INTEGER between 0 and 100.
        
        Output JSON:
        {{
            "brand_name": "{brand_name}",
            "confidence_score": 75,
            "contactibility_score": ...,
            "category_main_industry": "{{strategy_task.output.industry}}",
            "ai_reason_to_call": "{{strategy_task.output.ai_reason_to_call}}",
            "notes": "Extracted contacts. .ae fallback used if applicable.",
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
        context=[research_task, contact_task, strategy_task]
    )

    return Crew(
        agents=[researcher, contact_extractor, brand_strategist, analyst],
        tasks=[research_task, contact_task, strategy_task, validation_task],
        process=Process.sequential
    )
