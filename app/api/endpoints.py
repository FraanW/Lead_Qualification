import uuid
import os
import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from app.services.lead_processing import (
    jobs,
    process_excel_background,
    process_social_excel_background,
    process_business_excel_background,
)
from app.services.zoho import (
    get_access_token,
    search_duplicate,
    create_leads
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
        # Read the CSV as strings to preserve formatting and avoid 'nan' for empty cells
        df = pd.read_csv(file_path, dtype=str, na_filter=False)
        
        # Strip leading apostrophe from company_phone if column exists
        if 'company_phone' in df.columns:
            df['company_phone'] = df['company_phone'].str.lstrip("'")
            
        # Create an in-memory buffer
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type='text/csv',
            headers={"Content-Disposition": "attachment; filename=qualified_leads.csv"}
        )
    else:
        raise HTTPException(status_code=500, detail="File lost")


def map_rating(score, source):
    try:
        score = int(score)
    except:
        return "Cold"

    if score >= 90 or source=="news":
        return "Hot"
    elif score >= 70:
        return "Warm"
    else:
        return "Cold"

@router.post("/upload-leads")
async def upload_leads(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    # 1️⃣ Filter confidence > 70
    df = df[df["confidence_score"] > 70]

    # Replace NaN with empty string to avoid JSON errors
    df = df.fillna("")

    # 2️⃣ Remove duplicates inside CSV
    df = df.drop_duplicates(
        subset=["brand_name", "company_website", "company_email"]
    )

    access_token = await get_access_token()

    leads_to_create = []
    skipped_duplicates = []
    created_count = 0

    for _, row in df.iterrows():

        company_name = row.get("brand_name")
        website = row.get("company_website")
        email = row.get("company_email")

        # 3️⃣ Check duplicate in Zoho
        is_duplicate = await search_duplicate(
            access_token,
            company_name=company_name,
            website=website,
            email=email
        )

        if is_duplicate:
            skipped_duplicates.append(company_name)
            continue

        description = f"""
AI Reason:
{row.get("ai_reason_to_call", "")}

Notes:
{row.get("notes", "")}

Enrichment Status:
{row.get("enrichment_status", "")}

Confidence Score: {row.get("confidence_score", "")}
Contactibility Score: {row.get("contactibility_score", "")}
"""

        lead = {
            "Company": company_name,
            "Last_Name": company_name or "Unknown",  # 🔥 REQUIRED FIELD
            "Lead_Source": row.get("source") or "",
            "Phone": row.get("company_phone") or "",
            "Email": row.get("company_email") or "",
            "Website": row.get("company_website") or "",
            "Description": description,
            "Main_Industry": row.get("category_main_industry") or "",
            "Rating": map_rating(row.get("confidence_score"), row.get("source")),
        }

        leads_to_create.append(lead)

        # 4️⃣ Zoho batch limit = 100
        if len(leads_to_create) == 100:
            result = await create_leads(access_token, leads_to_create)

            for record in result.get("data", []):
                if record.get("status") == "success":
                    created_count += 1

            leads_to_create = []

    # Push remaining
    if leads_to_create:
        result = await create_leads(access_token, leads_to_create)

        for record in result.get("data", []):
            if record.get("status") == "success":
                created_count += 1

    return {
        "created": created_count,
        "skipped_duplicates": skipped_duplicates
    }