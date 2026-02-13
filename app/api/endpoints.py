import uuid
import os
import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from app.services.lead_processing import (
    jobs,
    process_excel_background,
    process_social_excel_background,
    process_business_excel_background,
)

router = APIRouter()


@router.post("/analyze-leads")
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


@router.post("/analyze-social-leads")
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


@router.post("/analyze_business_leads_post")
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


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/download/{job_id}")
async def download_results(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not ready or not found")
        
    file_path = job["output_file"]
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='text/csv', filename="qualified_leads.csv")
    else:
        raise HTTPException(status_code=500, detail="File lost")
