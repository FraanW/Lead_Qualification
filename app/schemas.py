from pydantic import BaseModel, Field
from typing import Optional, Literal

class CompanyDetails(BaseModel):
    phone: Optional[str] = ""
    email: Optional[str] = ""
    website: Optional[str] = ""
    Other: Optional[str] = ""

class DecisionMaker(BaseModel):
    name: Optional[str] = ""
    job_title: Optional[str] = ""
    mobile_number: Optional[str] = ""
    contact_number: Optional[str] = ""
    work_email: Optional[str] = ""

class LeadOutput(BaseModel):
    brand_name: str
    source: Literal["social", "news", "proximity"] = "news"
    category_main_industry: str = ""
    confidence_score: int = Field(default=0, ge=0, le=100)
    enrichment_status: Literal["enriched", "needs apollo"] = "needs apollo"
    company: CompanyDetails = Field(default_factory=CompanyDetails)
    decision_maker_1: DecisionMaker = Field(default_factory=DecisionMaker)
    ai_reason_to_call: str = ""
    notes: Optional[str] = ""

    class Config:
        json_schema_extra = {
            "example": {
                "brand_name": "Acme Corp",
                "source": "news",
                "category_main_industry": "Manufacturing",
                "confidence_score": 85,
                "enrichment_status": "needs apollo",
                "company": {
                    "phone": "123-456-7890",
                    "email": "info@acme.com",
                    "website": "www.acme.com",
                    "Other": ""
                },
                "decision_maker_1": {
                    "name": "", 
                    "job_title": "",
                    "mobile_number": "",
                    "contact_number": "",
                    "work_email": ""
                },
                "ai_reason_to_call": "Acme recently announced expansion...",
                "notes": "Good lead"
            }
        }
