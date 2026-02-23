import os
import httpx

ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")

ZOHO_TOKEN_URL = "https://accounts.zoho.in/oauth/v2/token"
ZOHO_BASE_URL = "https://www.zohoapis.in/crm/v2"


async def get_access_token():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            ZOHO_TOKEN_URL,
            params={
                "refresh_token": ZOHO_REFRESH_TOKEN,
                "client_id": ZOHO_CLIENT_ID,
                "client_secret": ZOHO_CLIENT_SECRET,
                "grant_type": "refresh_token",
            },
        )
        response.raise_for_status()
        return response.json()["access_token"]


async def search_duplicate(access_token, company_name=None, website=None, email=None):
    headers = {
        "Authorization": f"Zoho-oauthtoken {access_token}"
    }

    criteria_parts = []

    if company_name:
        criteria_parts.append(f"(Company:equals:{company_name})")

    if website:
        criteria_parts.append(f"(Website:equals:{website})")

    if email:
        criteria_parts.append(f"(Email:equals:{email})")

    if not criteria_parts:
        return False

    criteria = " or ".join(criteria_parts)

    url = f"{ZOHO_BASE_URL}/Leads/search?criteria={criteria}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

        if response.status_code == 204:
            return False  # No records found

        if response.status_code == 200:
            data = response.json()
            return len(data.get("data", [])) > 0

        return False


async def create_leads(access_token, leads):
    headers = {
        "Authorization": f"Zoho-oauthtoken {access_token}",
        "Content-Type": "application/json"
    }

    payload = {"data": leads}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ZOHO_BASE_URL}/Leads",
            headers=headers,
            json=payload
        )
        print("Zoho Status:", response.status_code)
        print("Zoho Body:", response.text)

        response.raise_for_status()
        return response.json()