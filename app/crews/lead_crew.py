from typing import Optional
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel

from app.core.config import Config
from app.tools.search_tool import SearXNGSearchTool
from app.tools.crawl_tool import WebCrawlTool


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

    # Logic to conditionally give search tool
    # If a specific website is provided, we remove the search tool to FORCE Scenario 1
    generic_domains = ["instagram", "facebook", "linkedin", "twitter", "tiktok", "youtube"]
    is_valid_website = website and not any(d in website.lower() for d in generic_domains)
    
    researcher_tools = [] if is_valid_website else [search_tool]

    researcher = Agent(
        role='Market Researcher',
        goal=f'Find and summarize the core business of {brand_name}',
        backstory="You are a business analyst. You analyze all companies objectively without political, social, or cultural bias. Your job is to find factual business information only.",
        tools=researcher_tools,
        llm=llm,
        verbose=True,
        max_iter=3,
        allow_delegation=False,
        max_retry_limit=1
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
        backstory="""You are a business analyst specializing in outdoor advertising. You analyze all companies objectively. 
        CRITICAL: Non-commercial entities like Government Authorities, Ministries, Customs, Police, and Public Spaces (malls, parks, beaches) are NOT target leads for standard private advertising. 
        You MUST penalize them with a confidence score of 0-10.""",
        llm=llm, 
        verbose=True,
        max_iter=2
    )

    research_task = Task(
        description=f"""
        Analyze the input website: "{website}"

        SCENARIO 1: Input website IS provided (e.g., "brand.com", "http://site.com")
        1. DO NOT call any tools.
        2. TRUST this website as the official one.
        3. Return the JSON below immediately.

        SCENARIO 2: Input website is EMPTY or GENERIC (e.g., "instagram.com", "facebook.com", "None")
        1. Search for '{brand_name} UAE official website' using the searxng_search tool.
        2. Analyze the result:
           - "Official Website (VERIFIED)": Use it.
           - "LOW CONFIDENCE": Use it but note risk.
           - "No official website found": Use empty string.

        CRITICAL OUTPUT RULES:
        - NEVER return a JSON with keys like "name", "parameters", "tool_output".
        - Your ONLY valid output is this JSON:
        
        {{
            "industry": "", 
            "website_url": "", 
            "local_contact_snippet": "",
            "notes": ""
        }}
        Fill "website_url" with the input website (Scenario 1) or search result (Scenario 2).
        If no results: {{ "industry": "Unknown", "website_url": "", "local_contact_snippet": "", "notes": "No data" }}
        """,
        expected_output="JSON object with industry, website_url, local_contact_snippet, notes",
        agent=researcher
    )

    # Task 2: Extract Contact Information
    contact_task = Task(
        description=f"""
        You will receive the research output from the previous task as context.
        
        1. Get "website_url" from research context.
        2. CHECK: Is "website_url" a valid URL (starts with http/https)?
           - YES: Call web_crawl(url=website_url).
           - NO: STOP. Do NOT call web_crawl. Return empty strings.

        3. Prioritize UAE/GCC contact details (+971 numbers and UAE addresses). 
        4. If the research context contains a local_contact_snippet, use that info.
        5. ONLY report contact info that was ACTUALLY found. Use empty string "" for anything not found.
        
        CRITICAL: 
        - DO NOT return a JSON with "name", "parameters", "web_crawl", or "tool_output". 
        - DO NOT return a simplified JSON like {{"url": "..."}}. You MUST use the full schema below.
        - DO NOT output the web_crawl tool signature.
        - You are now in DATA FORMATTING mode. Your ONLY valid output is the schema below.
        
        Respond with ONLY this JSON (no comments):
        {{
            "phone": "", 
            "email": "",
            "address": "",
            "other_contacts": ""
        }}
        Fill in only values you actually found. Do NOT invent data.
        """,
        expected_output="JSON with extracted contact details (UAE prioritized)",
        agent=contact_extractor,
        context=[research_task]
    )

    analysis_task = Task(
        description=f"""
        You will receive research and contact extraction results as context from previous tasks.
        Use the data from those contexts to fill in the output below.
        
        Final Qualification:
        1. Brand Verification: If the website_url does not match '{brand_name}', set confidence_score to 0.
        2. Penalize Govt/Public/Authority Entities: If the entity is a Government Authority (e.g., RTA, DEWA), Ministry, Customs, Police, or a Public Space (Mall/Beach/Park), you MUST use a confidence_score between 0 and 10.
        3. Prioritize UAE Presence: If both a global and a UAE contact (+971) were found, you MUST use the UAE one.
        4. Ensure the 'company' object uses the best UAE-specific contact info.
        5. Address formatting: Only prefix address in 'notes' if it exists.
        
        Scoring Guide (USE THIS to determine confidence_score — do NOT default to any number):
        - Commercial Retail/Auto/Healthcare/RealEstate: 70-90 (high billboard fit)
        - Commercial Fashion/Consumer brands: 60-80
        - B2B Manufacturing/Packaging: 20-40 (low fit)
        - Govt/State/Authorities (e.g. RTA)/Public Spaces: 0-10 (PENALIZED)
        - Unknown/No data/Brand Mismatch: 0
        
        MANDATORY CHECKLIST — every field MUST be filled:
        1. category_main_industry: Determine from the research context "industry" field. NEVER leave empty.
        2. confidence_score: Pick EXACTLY ONE number from the Scoring Guide ranges above based on the industry. Do NOT use 70, 80, or 85 as a lazy default.
        3. reason_to_call: Write a one-sentence reason why this brand is worth calling for OOH advertising.
        4. company.website: Use the website_url from the research task context.
        
        CRITICAL:
        - Output ONLY the JSON below. 
        - Do NOT include any intro/outro text like "Here is the JSON".
        - DO NOT output any tool signature or internal data.
        
        Output PURE JSON — no comments (no // or /* */), no trailing commas:
        {{
            "confidence_score": 0,
            "reason_to_call": "",
            "category_main_industry": "",
            "notes": "",
            "company": {{
                "phone": "",
                "email": "",
                "website": "",
                "Other": ""
            }}
        }}
        You MUST replace the 0 confidence_score with the correct value from the Scoring Guide. Fill ALL fields from context — no field may be empty if data exists in context.
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
