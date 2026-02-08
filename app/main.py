import asyncio
import uuid
import pandas as pd
import io
import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from .agents import get_lead_analysis_crew
from .schemas import LeadOutput, CompanyDetails, DecisionMaker
from .config import Config

app = FastAPI()

# In-memory storage for jobs
jobs = {}

# Semaphore to control concurrency
concurrency_limit = asyncio.Semaphore(Config.MAX_CONCURRENT_CREWS)

async def process_row(row: dict) -> dict:
    async with concurrency_limit:
        try:
            brand_name = row.get("Business Name")
            context = row.get("Context") or row.get("AI Reasoning", "")
            website = row.get("Website")
            
            # Simple clean up of "None" strings from Excel
            if str(website).lower() in ["none", "nan", ""]:
                website = None
            
            print(f"--- Processing: {brand_name} ---")
            
            # Kickoff the crew
            crew = get_lead_analysis_crew(brand_name, context, website)
            
            # Run blocking CrewAI call in thread
            result = await asyncio.to_thread(crew.kickoff)
            
            # Handle CrewOutput
            if hasattr(result, "pydantic") and result.pydantic:
                data = result.pydantic.dict()
            elif hasattr(result, "json_dict") and result.json_dict:
                data = result.json_dict
            else:
                # Fallback if parsing failed but we got text
                data = {
                    "website_url": website,
                    "industry": "Unknown",
                    "confidence_score": 0,
                    "reason_to_call": "AI Output Unstructured",
                    "notes": str(result)
                }
            
            # Return flat dict for CSV
            return {
                "brand_name": brand_name,
                "website_url": data.get("website_url"),
                "industry": data.get("industry"),
                "confidence_score": data.get("confidence_score"),
                "reason_to_call": data.get("reason_to_call"),
                "notes": data.get("notes")
            }
        except Exception as e:
            print(f"Error processing row {row.get('Business Name')}: {e}")
            return {
                "brand_name": row.get("Business Name", "Unknown"),
                "confidence_score": 0,
                "reason_to_call": "AI Analysis Failed",
                "notes": f"Error: {str(e)}",
                "website_url": row.get("Website")
            }

async def process_excel_background(job_id: str, df: pd.DataFrame):
    jobs[job_id]["status"] = "running"
    results = []
    
    # Create tasks
    tasks = []
    # Limit to first 5 rows for testing if needed, or process all. 
    # User said "thousands of rows", so we should process all, but carefully.
    # For this demo task, I'll process all.
    
    for i, row in df.iterrows():
        tasks.append(process_row(row.to_dict()))
    
    # execution
    processed_rows = await asyncio.gather(*tasks)
    
    # Flatten structure for CSV
    flattened_rows = []
    for r in processed_rows:
        # Flatten logic
        flat = r.copy()
        
        # Flatten company
        company = flat.pop("company", {})
        if company:
            for k, v in company.items():
                flat[f"company_{k}"] = v
                
        # Flatten decision_maker
        dm = flat.pop("decision_maker_1", {})
        if dm:
            for k, v in dm.items():
                flat[f"decision_maker_{k}"] = v
                
        flattened_rows.append(flat)

    # Save to CSV
    output_filename = f"processed_{job_id}.csv"
    out_df = pd.DataFrame(flattened_rows)
    out_df.to_csv(output_filename, index=False)
    
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["output_file"] = output_filename

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