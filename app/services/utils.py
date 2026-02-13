def clean_excel_value(val):
    if val is None:
        return ""
    s = str(val).strip()
    if s.startswith(('=', '@', '+', '-')) and len(s) > 0:
        return f"'{s}"  # Use single quote to force text in Excel/CSV
    return s

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
