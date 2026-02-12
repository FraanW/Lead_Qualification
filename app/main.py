import asyncio
import uuid
import pandas as pd
import io
import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from .agents import get_lead_analysis_crew, get_social_lead_analysis_crew, get_business_lead_analysis_crew
from .schemas import LeadOutput, CompanyDetails, DecisionMaker
from .config import Config

app = FastAPI()

# In-memory storage for jobs
jobs = {}

# Semaphore to control concurrency
concurrency_limit = asyncio.Semaphore(Config.MAX_CONCURRENT_CREWS)

def clean_excel_value(val):
    if val is None:
        return ""
    s = str(val).strip()
    if s.startswith(('=', '@', '+', '-')) and len(s) > 0:
        return f"'{s}"  # Use single quote to force text in Excel/CSV
    return s

async def process_row(row: dict) -> dict:
    async with concurrency_limit:
        try:
            brand_name = row.get("Business Name")
            
            # Skip empty/NaN rows
            if not brand_name or str(brand_name).lower() in ["nan", "none", "", "null"]:
                return None
            
            # Get Context or use Billboard-specific default
            context = row.get("Context") or row.get("AI Reasoning")
            if not context or str(context).lower() in ["nan", "none", "", "null"]:
                context = (
                    "Evaluate if this business is a good candidate for outdoor billboard advertising. "
                    "Look for B2C focus, local market presence, high customer lifetime value "
                    "(e.g., Real Estate, Legal, Home Services, Healthcare, Dealerships), "
                    "or brand awareness needs."
                )
            website = row.get("Website")
            
            # Simple clean up of "None" strings from Excel
            if str(website).lower() in ["none", "nan", ""]:
                website = None
            
            print(f"--- Processing: {brand_name} ---")
            
            # Kickoff the crew
            crew = get_lead_analysis_crew(brand_name, context, website)
            
            # Run blocking CrewAI call in thread with timeout
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(crew.kickoff), 
                    timeout=150  # Increased for deeper research
                )
            except asyncio.TimeoutError:
                # Calculate contactability: 30 if website available, 0 otherwise
                timeout_contactibility = 30 if website else 0
                lead = LeadOutput(
                    brand_name=brand_name,
                    source="news",
                    category_main_industry="Error",
                    confidence_score=0,
                    contactibility_score=timeout_contactibility,
                    enrichment_status="needs apollo",
                    company=CompanyDetails(
                        phone="",
                        email="",
                        website=website or "",
                        Other=""
                    ),
                    decision_maker_1=DecisionMaker(
                        name="",
                        job_title="",
                        mobile_number="",
                        contact_number="",
                        work_email=""
                    ),
                    ai_reason_to_call="Timeout - manual review",
                    notes="Processing took longer than 30s"
                )
                return lead.model_dump()
            
            # Handle CrewOutput - extract JSON from raw string
            try:
                import json
                import re
                
                # Get the raw output as string
                raw_output = str(result.raw) if hasattr(result, 'raw') else str(result)
                
                # Try to find JSON in the output (handles both raw JSON and text with JSON)
                json_match = re.search(r'\{.*"confidence_score".*\}', raw_output, re.DOTALL)
                
                if json_match:
                    data = json.loads(json_match.group(0))
                else:
                    # Fallback if no JSON found
                    print(f"No JSON found in output for {brand_name}, raw: {raw_output[:200]}")
                    data = {
                        "category_main_industry": "Unknown",
                        "confidence_score": 0,
                        "reason_to_call": "AI Output Unstructured",
                        "notes": raw_output[:500],
                        "company": {
                            "phone": "",
                            "email": "",
                            "website": website or "",
                            "Other": ""
                        }
                    }
            except Exception as e:
                # Nuclear option - force valid data if parsing crashes
                print(f"Parsing error for {brand_name}: {e}")
                data = {
                    "category_main_industry": "Unknown", 
                    "confidence_score": 0, 
                    "reason_to_call": "Schema validation failed", 
                    "notes": "Fallback activated due to malformed AI output",
                    "company": {
                        "phone": "",
                        "email": "",
                        "website": website or "",
                        "Other": ""
                    }
                }
            
            # Extract company info with fallbacks
            company_data = data.get("company", {})
            if not isinstance(company_data, dict):
                company_data = {}
            
            company_phone = company_data.get("phone", "")
            company_email = company_data.get("email", "")
            company_website = company_data.get("website", "") or data.get("website_url", "") or website or ""
            company_other = company_data.get("Other", "")
            
            # Calculate contactability score
            # 100: all 3 (website + email + phone)
            # 60: 2 available
            # 30: 1 available
            # 0: none available
            has_website = bool(company_website)
            has_email = bool(company_email)
            has_phone = bool(company_phone)
            
            available_count = sum([has_website, has_email, has_phone])
            
            if available_count == 3:
                contactibility_score = 100
            elif available_count == 2:
                contactibility_score = 60
            elif available_count == 1:
                contactibility_score = 30
            else:
                contactibility_score = 0
            
            # Robust Fallback (User Request)
            category = data.get("category_main_industry") or data.get("industry", "Unknown")
            if not category or category == "Unknown":
                # If AI returned nothing useful or explicitly unknown, ensure we have defaults
                if not data.get("reason_to_call"):
                    data["reason_to_call"] = "Insufficient data for detailed pitch"
                if not data.get("confidence_score"):
                    data["confidence_score"] = 0
            
            # Return LeadOutput structure for consistency with social leads
            lead = LeadOutput(
                brand_name=brand_name,
                source="news",
                category_main_industry=category,
                confidence_score=data.get("confidence_score", 0),
                contactibility_score=contactibility_score,
                enrichment_status="needs apollo",
                company=CompanyDetails(
                    phone=company_phone,
                    email=company_email,
                    website=company_website,
                    Other=company_other
                ),
                decision_maker_1=DecisionMaker(
                    name="",
                    job_title="",
                    mobile_number="",
                    contact_number="",
                    work_email=""
                ),
                ai_reason_to_call=data.get("reason_to_call", ""),
                notes=data.get("notes", "")
            )
            return lead.model_dump()
        except Exception as e:
            print(f"Error processing row {brand_name}: {e}")
            # Calculate contactability: 30 if website available, 0 otherwise
            error_contactibility = 30 if website else 0
            lead = LeadOutput(
                brand_name=brand_name,
                source="news",
                category_main_industry="Error",
                confidence_score=0,
                contactibility_score=error_contactibility,
                enrichment_status="needs apollo",
                company=CompanyDetails(
                    phone="",
                    email="",
                    website=website or "",
                    Other=""
                ),
                decision_maker_1=DecisionMaker(
                    name="",
                    job_title="",
                    mobile_number="",
                    contact_number="",
                    work_email=""
                ),
                ai_reason_to_call="Processing failed - manual review needed",
                notes=f"Error: {str(e)}"
            )
            return lead.model_dump()

async def process_excel_background(job_id: str, df: pd.DataFrame):
    jobs[job_id]["status"] = "running"
    results = []
    
    # Remove empty rows and deduplicate before processing
    df = df.dropna(subset=["Business Name"])
    df = df.drop_duplicates(subset=["Business Name"], keep='first')
    
    # Create tasks
    tasks = []
    # Limit to first 5 rows for testing if needed, or process all. 
    # User said "thousands of rows", so we should process all, but carefully.
    # For this demo task, I'll process all.
    
    for i, row in df.iterrows():
        tasks.append(process_row(row.to_dict()))
    
    # execution
    processed_rows = await asyncio.gather(*tasks)
    
    # Filter out None results (skipped rows)
    processed_rows = [r for r in processed_rows if r is not None]
    
    # Flatten structure for CSV - matching social leads format exactly
    flattened_rows = []
    for r in processed_rows:
        flat = {}
        # Manually flatten in the exact order as social leads
        flat["brand_name"] = clean_excel_value(r.get("brand_name"))
        flat["source"] = clean_excel_value(r.get("source"))
        flat["category_main_industry"] = clean_excel_value(r.get("category_main_industry"))
        flat["confidence_score"] = r.get("confidence_score")
        flat["contactibility_score"] = r.get("contactibility_score")
        flat["enrichment_status"] = clean_excel_value(r.get("enrichment_status"))
        
        company = r.get("company", {})
        flat["company_phone"] = clean_excel_value(company.get("phone"))
        flat["company_email"] = clean_excel_value(company.get("email"))
        flat["company_website"] = clean_excel_value(company.get("website"))
        flat["company_other"] = clean_excel_value(company.get("Other"))
        
        dm = r.get("decision_maker_1", {})
        flat["dm_name"] = clean_excel_value(dm.get("name"))
        flat["dm_job_title"] = clean_excel_value(dm.get("job_title"))
        flat["dm_mobile"] = clean_excel_value(dm.get("mobile_number"))
        flat["dm_contact"] = clean_excel_value(dm.get("contact_number"))
        flat["dm_email"] = clean_excel_value(dm.get("work_email"))
        
        flat["ai_reason_to_call"] = clean_excel_value(r.get("ai_reason_to_call"))
        flat["notes"] = clean_excel_value(r.get("notes"))
        
        flattened_rows.append(flat)

    # Save to CSV
    output_filename = f"processed_{job_id}.csv"
    pd.DataFrame(flattened_rows).to_csv(output_filename, index=False)
    
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["output_file"] = output_filename

async def process_social_row(row: dict, is_retry: bool = False) -> dict:
    async with concurrency_limit:
        try:
            brand_name = row.get("Brand")
            if not brand_name or str(brand_name).lower() in ["nan", "none", "", "null"]:
                return None
            
            post_reason = row.get("Reason to call for OOH")
            influencer = row.get("Influencer promoting")
            website = row.get("Website")
            email = row.get("Email")
            
            # Simple clean up
            if str(website).lower() in ["none", "nan", ""]: website = None
            if str(email).lower() in ["none", "nan", ""]: email = None
            
            print(f"--- Processing Social Lead{' (RETRY)' if is_retry else ''}: {brand_name} ---")
            
            # Kickoff the social crew
            crew = get_social_lead_analysis_crew(brand_name, influencer, post_reason, website)
            
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(crew.kickoff), 
                    timeout=150 if is_retry else 120 
                )
            except asyncio.TimeoutError:
                return {
                    "data": LeadOutput(
                        brand_name=brand_name,
                        source="social",
                        ai_reason_to_call="Timeout during research",
                        notes="Processing exceeded time limit"
                    ).model_dump(),
                    "needs_retry": not is_retry and website is not None
                }
            
            # Extract JSON
            try:
                import json
                import re
                raw_output = str(result.raw) if hasattr(result, 'raw') else str(result)
                json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
                
                if json_match:
                    data = json.loads(json_match.group(0))
                    
                    # Clean up "..." artifacts if present
                    def clean(val):
                        return "" if val == "..." or val == "Not Found" else val

                    c_phone = clean(data.get("company", {}).get("phone", ""))
                    c_email = clean(data.get("company", {}).get("email", email or ""))
                    c_website = clean(data.get("company", {}).get("website", website or ""))
                    c_other = clean(data.get("company", {}).get("Other", ""))
                    
                    # Calculate contactability score
                    has_website = bool(c_website)
                    has_email = bool(c_email)
                    has_phone = bool(c_phone)
                    
                    available_count = sum([has_website, has_email, has_phone])
                    
                    if available_count == 3:
                        contactibility_score = 100
                    elif available_count == 2:
                        contactibility_score = 60
                    elif available_count == 1:
                        contactibility_score = 30
                    else:
                        contactibility_score = 0

                    # Check retry condition (score 0 means likely failed research)
                    should_retry = False
                    if not is_retry and website and contactibility_score == 0:
                        should_retry = True
                    
                    # Map back to LeadOutput
                    lead = LeadOutput(
                        brand_name=brand_name,
                        source="social",
                        category_main_industry=clean(data.get("category_main_industry", "Unknown")),
                        confidence_score=data.get("confidence_score", 0),
                        contactibility_score=contactibility_score,
                        enrichment_status="needs apollo",
                        company=CompanyDetails(
                            phone=c_phone,
                            email=c_email,
                            website=c_website,
                            Other=c_other
                        ),
                        ai_reason_to_call=clean(data.get("ai_reason_to_call", "")),
                        notes=clean(data.get("notes", ""))
                    )
                    return {"data": lead.model_dump(), "needs_retry": should_retry, "original_row": row}
                else:
                    raise ValueError("No JSON found")
            except Exception as e:
                print(f"Parsing error for {brand_name}: {e}")
                return {
                    "data": LeadOutput(
                        brand_name=brand_name,
                        source="social",
                        ai_reason_to_call="AI output parsing failed",
                        notes=f"Raw: {raw_output[:200]}"
                    ).model_dump(),
                    "needs_retry": not is_retry and website is not None,
                    "original_row": row
                }
                
        except Exception as e:
            print(f"Error processing social row {brand_name}: {e}")
            return {
                "data": LeadOutput(
                    brand_name=brand_name,
                    source="social",
                    notes=f"Error: {str(e)}"
                ).model_dump(),
                "needs_retry": False
            }

def normalize_phone(phone: str) -> str:
    """Normalize phone number for comparison by removing spaces, dashes, parentheses"""
    if not phone:
        return ""
    # Remove common separators and spaces
    normalized = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace("+", "")
    return normalized.lower()

def phones_match(phone1: str, phone2: str) -> bool:
    """Check if two phone numbers match after normalization"""
    if not phone1 or not phone2:
        return False
    norm1 = normalize_phone(phone1)
    norm2 = normalize_phone(phone2)
    # Check if one contains the other (handles country code differences)
    return norm1 in norm2 or norm2 in norm1

async def process_business_row(row: dict) -> dict:
    """Process business lead with contact validation and penalty scoring"""
    async with concurrency_limit:
        try:
            brand_name = row.get("Brand")
            if not brand_name or str(brand_name).lower() in ["nan", "none", "", "null"]:
                return None
            
            # Extract input data
            input_phone = row.get("Contact", "")
            input_email = row.get("Email", "")
            website = row.get("Website", "")
            
            # Clean up
            if str(input_phone).lower() in ["none", "nan", ""]: input_phone = ""
            if str(input_email).lower() in ["none", "nan", ""]: input_email = ""
            if str(website).lower() in ["none", "nan", ""]: website = None
            
            print(f"--- Processing Business Lead: {brand_name} ---")
            print(f"    Input Phone: {input_phone}, Input Email: {input_email}")
            
            # Kickoff the crew
            crew = get_business_lead_analysis_crew(brand_name, website)
            
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(crew.kickoff), 
                    timeout=150
                )
            except asyncio.TimeoutError:
                contactibility = 30 if website else 0
                return LeadOutput(
                    brand_name=brand_name,
                    source="business",
                    contactibility_score=contactibility,
                    ai_reason_to_call="Timeout during research",
                    notes="Processing exceeded time limit"
                ).model_dump()
            
            # Extract JSON
            try:
                import json
                import re
                raw_output = str(result.raw) if hasattr(result, 'raw') else str(result)
                json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
                
                if json_match:
                    data = json.loads(json_match.group(0))
                    
                    # Clean up "..." artifacts
                    def clean(val):
                        return "" if val == "..." or val == "Not Found" else val

                    # Extract fetched contact info
                    company_data = data.get("company", {})
                    fetched_phone = clean(company_data.get("phone", ""))
                    fetched_email = clean(company_data.get("email", ""))
                    fetched_website = clean(company_data.get("website", "") or data.get("official_website", "") or website or "")
                    fetched_other = clean(company_data.get("Other", ""))
                    
                    # Contact validation and penalty logic
                    penalty = 0
                    final_phone = fetched_phone
                    final_email = fetched_email
                    
                    # Phone validation
                    if input_phone and fetched_phone:
                        if phones_match(input_phone, fetched_phone):
                            # Match - use fetched (more complete with country code)
                            final_phone = fetched_phone
                        else:
                            # Mismatch - merge and apply penalty
                            final_phone = f"{input_phone}, {fetched_phone}"
                            penalty += 10
                            print(f"    Phone mismatch! Input: {input_phone}, Fetched: {fetched_phone}, Penalty: -10")
                    elif input_phone and not fetched_phone:
                        # Use input phone if nothing fetched
                        final_phone = input_phone
                    
                    # Email validation
                    if input_email and fetched_email:
                        if input_email.lower().strip() == fetched_email.lower().strip():
                            # Match - use fetched
                            final_email = fetched_email
                        else:
                            # Mismatch - merge and apply penalty
                            final_email = f"{input_email}, {fetched_email}"
                            penalty += 10
                            print(f"    Email mismatch! Input: {input_email}, Fetched: {fetched_email}, Penalty: -10")
                    elif input_email and not fetched_email:
                        # Use input email if nothing fetched
                        final_email = input_email
                    
                    # Calculate base contactability score
                    has_website = bool(fetched_website)
                    has_email = bool(final_email)
                    has_phone = bool(final_phone)
                    
                    available_count = sum([has_website, has_email, has_phone])
                    
                    if available_count == 3:
                        base_score = 100
                    elif available_count == 2:
                        base_score = 60
                    elif available_count == 1:
                        base_score = 30
                    else:
                        base_score = 0
                    
                    # Apply penalty
                    final_contactibility_score = max(0, base_score - penalty)
                    
                    print(f"    Base Score: {base_score}, Penalty: {penalty}, Final: {final_contactibility_score}")
                    
                    # Build LeadOutput
                    lead = LeadOutput(
                        brand_name=brand_name,
                        source="business",
                        category_main_industry=data.get("category_main_industry", "Unknown"),
                        confidence_score=data.get("confidence_score", 0),
                        contactibility_score=final_contactibility_score,
                        enrichment_status="needs apollo",
                        company=CompanyDetails(
                            phone=final_phone,
                            email=final_email,
                            website=fetched_website,
                            Other=fetched_other
                        ),
                        ai_reason_to_call=data.get("ai_reason_to_call", ""),
                        notes=data.get("notes", "")
                    )
                    return lead.model_dump()
                else:
                    raise ValueError("No JSON found")
            except Exception as e:
                print(f"Parsing error for {brand_name}: {e}")
                return LeadOutput(
                    brand_name=brand_name,
                    source="business",
                    ai_reason_to_call="AI output parsing failed",
                    notes=f"Raw: {raw_output[:200]}"
                ).model_dump()
                
        except Exception as e:
            print(f"Error processing business row {brand_name}: {e}")
            return LeadOutput(
                brand_name=brand_name,
                source="business",
                notes=f"Error: {str(e)}"
            ).model_dump()

async def process_social_excel_background(job_id: str, df: pd.DataFrame):
    jobs[job_id]["status"] = "running"
    
    # Remove empty rows and deduplicate
    df = df.dropna(subset=["Brand"])
    df = df.drop_duplicates(subset=["Brand"], keep='first')
    
    # First Pass
    tasks = []
    for i, row in df.iterrows():
        tasks.append(process_social_row(row.to_dict()))
    
    results = await asyncio.gather(*tasks)
    
    final_results = []
    retry_tasks = []
    
    for res in results:
        if res is None: continue
        
        if res.get("needs_retry"):
            print(f"Queueing retry for: {res['data']['brand_name']}")
            retry_tasks.append(process_social_row(res["original_row"], is_retry=True))
        else:
            final_results.append(res["data"])
            
    # Second Pass (Retry once)
    if retry_tasks:
        print(f"Starting retry pass for {len(retry_tasks)} leads...")
        retry_results = await asyncio.gather(*retry_tasks)
        for r in retry_results:
            if r:
                final_results.append(r["data"])
    
    # Flatten
    flattened_rows = []
    for r in final_results:
        flat = {}
        # Manually flatten with Excel cleaning
        flat["brand_name"] = clean_excel_value(r.get("brand_name"))
        flat["source"] = clean_excel_value(r.get("source"))
        flat["category_main_industry"] = clean_excel_value(r.get("category_main_industry"))
        flat["confidence_score"] = r.get("confidence_score")
        flat["contactibility_score"] = r.get("contactibility_score")
        flat["enrichment_status"] = clean_excel_value(r.get("enrichment_status"))
        
        company = r.get("company", {})
        flat["company_phone"] = clean_excel_value(company.get("phone"))
        flat["company_email"] = clean_excel_value(company.get("email"))
        flat["company_website"] = clean_excel_value(company.get("website"))
        flat["company_other"] = clean_excel_value(company.get("Other"))
        
        dm = r.get("decision_maker_1", {})
        flat["dm_name"] = clean_excel_value(dm.get("name"))
        flat["dm_job_title"] = clean_excel_value(dm.get("job_title"))
        flat["dm_mobile"] = clean_excel_value(dm.get("mobile_number"))
        flat["dm_contact"] = clean_excel_value(dm.get("contact_number"))
        flat["dm_email"] = clean_excel_value(dm.get("work_email"))
        
        flat["ai_reason_to_call"] = clean_excel_value(r.get("ai_reason_to_call"))
        flat["notes"] = clean_excel_value(r.get("notes"))
        
        flattened_rows.append(flat)

    output_filename = f"social_processed_{job_id}.csv"
    pd.DataFrame(flattened_rows).to_csv(output_filename, index=False)
    
    jobs[job_id]["output_file"] = output_filename
    jobs[job_id]["status"] = "completed"

@app.post("/analyze-social-leads")
async def analyze_social_leads(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "queued"}
        
        background_tasks.add_task(process_social_excel_background, job_id, df)
        
        return {"job_id": job_id, "message": "Social media file uploaded. Processing started.", "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_business_excel_background(job_id: str, df: pd.DataFrame):
    """Background task to process business leads with contact validation"""
    jobs[job_id]["status"] = "running"
    
    # Remove empty rows and deduplicate
    df = df.dropna(subset=["Brand"])
    df = df.drop_duplicates(subset=["Brand"], keep='first')
    
    # Process all rows
    tasks = [process_business_row(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Flatten to CSV format (same as social leads)
    flattened_rows = []
    for lead_data in results:
        flat = {
            "brand_name": clean_excel_value(lead_data.get("brand_name", "")),
            "source": clean_excel_value(lead_data.get("source", "business")),
            "category_main_industry": clean_excel_value(lead_data.get("category_main_industry", "")),
            "confidence_score": lead_data.get("confidence_score", 0),
            "contactibility_score": lead_data.get("contactibility_score", 0),
            "enrichment_status": clean_excel_value(lead_data.get("enrichment_status", "needs apollo")),
            "company_phone": clean_excel_value(lead_data.get("company", {}).get("phone", "")),
            "company_email": clean_excel_value(lead_data.get("company", {}).get("email", "")),
            "company_website": clean_excel_value(lead_data.get("company", {}).get("website", "")),
            "company_other": clean_excel_value(lead_data.get("company", {}).get("Other", "")),
            "dm_name": clean_excel_value(lead_data.get("decision_maker_1", {}).get("name", "")),
            "dm_job_title": clean_excel_value(lead_data.get("decision_maker_1", {}).get("job_title", "")),
            "dm_mobile": clean_excel_value(lead_data.get("decision_maker_1", {}).get("mobile_number", "")),
            "dm_contact": clean_excel_value(lead_data.get("decision_maker_1", {}).get("contact_number", "")),
            "dm_email": clean_excel_value(lead_data.get("decision_maker_1", {}).get("work_email", "")),
            "ai_reason_to_call": clean_excel_value(lead_data.get("ai_reason_to_call", "")),
            "notes": clean_excel_value(lead_data.get("notes", ""))
        }
        flattened_rows.append(flat)

    output_filename = f"business_processed_{job_id}.csv"
    pd.DataFrame(flattened_rows).to_csv(output_filename, index=False)
    
    jobs[job_id]["output_file"] = output_filename
    jobs[job_id]["status"] = "completed"

@app.post("/analyze_business_leads_post")
async def analyze_business_leads_post(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Endpoint to process business leads with contact validation.
    Accepts CSV or Excel with columns: Brand, Reason to call for OOH, Contact, Email, Website, Category, Address, Notes, Extra notes
    """
    try:
        contents = await file.read()
        
        # Detect file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="File must be CSV or Excel (.csv, .xlsx, .xls)")
        
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "queued"}
        
        background_tasks.add_task(process_business_excel_background, job_id, df)
        
        return {"job_id": job_id, "message": "Business leads file uploaded. Processing started.", "rows": len(df)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze-leads")
async def analyze_leads(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Handle Excel
        df = pd.read_excel(io.BytesIO(contents))
        
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "queued"}
        
        background_tasks.add_task(process_excel_background, job_id, df)
        
        return {"job_id": job_id, "message": "File uploaded. Processing started.", "rows": len(df)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/download/{job_id}")
async def download_results(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not ready or not found")
        
    file_path = job["output_file"]
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='text/csv', filename="qualified_leads.csv")
    else:
        raise HTTPException(status_code=500, detail="File lost")