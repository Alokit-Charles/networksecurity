import re
from urllib.parse import urlparse
import pandas as pd

SHORTENERS = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.ly"

# Exact 30 feature schema matching your model training fit order
FEATURES = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL",
    "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
    "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
    "Statistical_report"
]

def extract_features_from_url(url: str) -> pd.DataFrame:
    """Extracts structural features from a single raw URL matching exact schema case and order."""
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    parsed = urlparse(url)
    domain = parsed.netloc

    features = {}

    # 1. having_IP_Address
    ip_pattern = r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])|0x[0-9a-fA-F]+"
    features['having_IP_Address'] = -1 if re.search(ip_pattern, url) else 1

    # 2. URL_Length
    url_len = len(url)
    if url_len < 54:
        features['URL_Length'] = 1
    elif 54 <= url_len <= 75:
        features['URL_Length'] = 0
    else:
        features['URL_Length'] = -1

    # 3. Shortining_Service
    features['Shortining_Service'] = -1 if re.search(SHORTENERS, url, flags=re.IGNORECASE) else 1

    # 4. having_At_Symbol
    features['having_At_Symbol'] = -1 if "@" in url else 1

    # 5. double_slash_redirecting
    features['double_slash_redirecting'] = -1 if url.rfind("//") > 7 else 1

    # 6. Prefix_Suffix
    features['Prefix_Suffix'] = -1 if "-" in domain else 1

    # 7. having_Sub_Domain
    clean_domain = re.sub(r"^www\.", "", domain)
    dot_count = clean_domain.count(".")
    if dot_count <= 1:
        features['having_Sub_Domain'] = 1
    elif dot_count == 2:
        features['having_Sub_Domain'] = -1
    else:
        features['having_Sub_Domain'] = -1

    # 8. SSLfinal_State
    features['SSLfinal_State'] = 1 if parsed.scheme == 'https' else -1

    # 12. HTTPS_token (EXACT match with upper-case HTTPS)
    features['HTTPS_token'] = -1 if 'https' in domain.lower() else 1

    # Populate default values (1) for non-URL-parsable DOM features
    for col in FEATURES:
        if col not in features:
            features[col] = -1

    # Convert to DataFrame and enforce exact column ordering
    df = pd.DataFrame([features])
    return df[FEATURES]