from crewai import Agent, Task, Crew, Process, LLM

from app.core.config import Config
from app.tools.search_tool import SearXNGSearchTool
from app.tools.crawl_tool import WebCrawlTool


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

    # Logic to conditionally give search tool
    # If a specific website is provided, we remove the search tool to FORCE Scenario 1
    generic_domains = ["instagram", "facebook", "linkedin", "twitter", "tiktok", "youtube"]
    is_valid_website = website and not any(d in website.lower() for d in generic_domains)
    
    researcher_tools = [] if is_valid_website else [search_tool]

    researcher = Agent(
        role='Business Intelligence Researcher',
        goal=f'Find and verify the official website of {brand_name}. Focus on UAE presence.',
        backstory="You are experts at identifying the ONE official website for a brand, avoiding social media pages or directories. You prioritize UAE-specific domains (.ae) if they exist.",
        tools=researcher_tools,
        llm=llm,
        verbose=True,
        max_iter=3,
        allow_delegation=False,
        max_retry_limit=1
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
        backstory="""You synthesize all data. You analyze companies objectively. 
        CRITICAL: Non-commercial entities like Government Authorities (e.g., RTA, DEWA, Municipality), State-owned non-commercial entities, Ministries, Customs, Police, and Public Spaces (malls, parks, beaches) are NOT target leads for standard private advertising. 
        You MUST penalize them with a confidence_score of 0-10. These are public services, not commercial products/services suitable for OOH billboards.""",
        llm=llm, 
        verbose=True,
        max_iter=2
    )

    # Task 1: Find Official Website with .ae Fallback
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
            "brand_name": "{brand_name}",
            "official_website": "", 
            "local_contact_snippet": "",
            "industry_guess": ""
        }}
        Fill "official_website" with the input website (Scenario 1) or search result (Scenario 2).
        """,
        expected_output="JSON with official website URL and local contact snippet",
        agent=researcher
    )

    # Task 2: Targeted Contact Extraction
    contact_task = Task(
        description=f"""
        You will receive the research output from the previous task as context.
        
        1. Get "official_website" from research context.
        2. CHECK: Is "official_website" a valid URL (starts with http/https)?
           - YES: Call web_crawl(url=official_website).
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

    # Task 3: Brand Strategy
    strategy_task = Task(
        description=f"""
        You will receive the research and contact extraction results as context from previous tasks.
        
        1. ONLY call web_crawl if the official website URL starts with http:// or https://. Otherwise skip.
        2. Craft a ONE SENTENCE 'reason to call' for {brand_name} about outdoor advertising.
        3. Identify their main industry (e.g., "Food and Beverage", "Automotive", "Fashion", "Healthcare", etc.). NEVER leave empty.
        
        CRITICAL:
        - Output ONLY the JSON below. 
        - Do NOT include any intro/outro text like "Here is the JSON".
        - Do NOT output the web_crawl tool signature.
        - DO NOT output a JSON with "name": "web_crawl".
        
        Respond with ONLY this JSON (no comments):
        {{
            "ai_reason_to_call": "",
            "industry": ""
        }}
        Fill in both fields. If industry is unknown, use "Unknown". Do not leave empty.
        If web_crawl fails, use the best info you have.
        """,
        expected_output="JSON with AI reason to call and industry",
        agent=brand_strategist,
        context=[research_task, contact_task]
    )

    # Task 4: Final Validation and Scoring
    validation_task = Task(
        description=f"""
        You will receive research, contact, and strategy results as context from previous tasks.
        Use the data from those contexts to fill in the output below.
        
        Final Qualification:
        1. Brand Verification: If the official_website does not match '{brand_name}', set confidence_score to 0.
        2. Penalize Govt/Public/Authority Entities: If the entity is a Government Authority (e.g., RTA, DEWA), Ministry, Customs, Police, or a Public Space (Mall/Beach/Park), you MUST use a confidence_score between 0 and 10.
        3. Prioritize UAE Presence: If both a global and a UAE contact (+971) were found, you MUST use the UAE one.
        4. Ensure the 'company' object uses the best UAE-specific contact info.
        5. Address formatting: Only prefix address in 'notes' if it exists.
        
        Scoring Guide (USE THIS to determine confidence_score — do NOT default to any number):
        - Commercial Retail/Auto/Healthcare/RealEstate: 70-90 (high billboard fit)
        - Commercial Fashion/Consumer brands: 60-80
        - B2B Manufacturing/Packaging: 20-40 (low fit)
        - Govt/State Authorities (e.g. RTA)/Public Spaces: 0-10 (PENALIZED)
        - Unknown/No data/Brand Mismatch: 0
        
        MANDATORY CHECKLIST — every field MUST be filled:
        1. category_main_industry: Get the "industry" from the strategy task context. NEVER leave empty.
        2. confidence_score: Pick EXACTLY ONE number from the Scoring Guide ranges above based on the industry. Do NOT use 70, 80, or 85 as a lazy default.
        3. ai_reason_to_call: Get the "ai_reason_to_call" from the strategy task context.
        4. company.website: Use the official_website from the research task context.
        
        Output PURE JSON — no comments (no // or /* */), no trailing commas:
        {{
            "brand_name": "{brand_name}",
            "confidence_score": 0,
            "contactibility_score": 0,
            "category_main_industry": "",
            "ai_reason_to_call": "",
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
        expected_output="Final Lead JSON",
        agent=analyst,
        context=[research_task, contact_task, strategy_task]
    )

    return Crew(
        agents=[researcher, contact_extractor, brand_strategist, analyst],
        tasks=[research_task, contact_task, strategy_task, validation_task],
        process=Process.sequential
    )
