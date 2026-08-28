"""
Full feature extractor for the Mohammad, Thabtah & McCluskey
"Phishing Websites Features" scheme (the classic UCI Phishing Websites
dataset). Implements all 30 features, in the exact order the original
dataset.

Features 26-30 originally depended on services that either no longer
exist or are now paywalled:
  - web_traffic          : Alexa (used in the paper) was shut down by
                            Amazon in 2022. Substituted with the Tranco
                            list (https://tranco-list.eu), the standard
                            academic replacement for Alexa rankings.
  - Page_Rank             : Google's public PageRank API was discontinued
                            in 2016 and never had a real replacement.
                            No free live source exists.
  - Google_Index           : Scraping Google search results violates
                            Google's ToS and is unreliable, so this is
                            NOT done by screen-scraping. If you have a
                            Google Programmable Search Engine API key,
                            plug it into `google_index()` -- instructions
                            are inline. Otherwise defaults to "assume
                            indexed" (1), which is what the paper's rule
                            effectively defaults new/legitimate sites to.
  - Links_pointing_to_page : Needs a backlink index (Moz/Ahrefs/Majestic),
                              all paid APIs. Defaults to a neutral value
                              unless you supply an API key.
  - Statistical_report     : Checks the URL/domain/IP against PhishTank's
                              live public database (free, no key required
                              for basic lookups) instead of the paper's
                              static 2010-2012 top-10 lists.
"""

import re
import socket
import ssl
import time
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import tldextract
    _HAS_TLDEXTRACT = True
except ImportError:
    _HAS_TLDEXTRACT = False

try:
    import whois as pywhois
    _HAS_WHOIS = True
except ImportError:
    _HAS_WHOIS = False

try:
    import dns.resolver
    _HAS_DNSPYTHON = True
except ImportError:
    _HAS_DNSPYTHON = False


# Exact 30 feature schema, matching your model's training fit order
FEATURES = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL",
    "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
    "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
    "Statistical_report",
]

SHORTENERS = (
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|"
    r"cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|"
    r"snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|"
    r"snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|"
    r"om\.ly|to\.ly|bit\.do|t\.ly"
)

TRUSTED_ISSUERS = [
    # From the paper (2015-era list)
    "geotrust", "godaddy", "network solutions", "thawte", "comodo", "doster", "verisign",
    # Modern additions -- the paper's list is a decade out of date, and most
    # legitimate sites today use these instead
    "digicert", "let's encrypt", "sectigo", "globalsign", "amazon", "google trust services",
    "microsoft", "cloudflare", "ssl.com", "entrust", "identrust",
]

# Ports and whether they should be OPEN (per Table 1 in the paper)
PORT_TABLE = {
    21: False,   # FTP -> should be closed
    22: False,   # SSH -> should be closed
    23: False,   # Telnet -> should be closed
    80: True,    # HTTP -> should be open
    443: True,   # HTTPS -> should be open
    445: False,  # SMB -> should be closed
    1433: False, # MSSQL -> should be closed
    1521: False, # ORACLE -> should be closed
    3306: False, # MySQL -> should be closed
    3389: False, # Remote Desktop -> should be closed
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 8


# ------------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def _registrable_domain(netloc: str) -> str:
    """Best-effort second-level+public-suffix domain, e.g. 'sub.hud.ac.uk' -> 'hud.ac.uk'."""
    host = netloc.split(":")[0]
    if _HAS_TLDEXTRACT:
        ext = tldextract.extract(host)
        return ".".join(part for part in [ext.domain, ext.suffix] if part)
    # Fallback: strip a small set of common two-part ccTLDs, else last two labels
    two_part_suffixes = {
        "co.uk", "ac.uk", "gov.uk", "org.uk", "me.uk", "net.uk",
        "co.jp", "co.in", "co.nz", "co.za", "co.kr", "com.au", "net.au",
        "com.br", "com.cn", "com.mx", "com.tr", "com.sg",
    }
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in two_part_suffixes:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def _fetch(url: str):
    """Fetch a URL, following redirects. Returns (response_or_None, html_or_None, redirect_count)."""
    try:
        resp = requests.get(
            url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True
        )
        return resp, resp.text, len(resp.history)
    except requests.RequestException:
        return None, None, 0


def _get_whois(domain: str):
    if not _HAS_WHOIS:
        return None
    try:
        return pywhois.whois(domain)
    except Exception:
        return None


def _first_date(value):
    """python-whois sometimes returns a list of dates; normalize to one datetime."""
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value
    return None


# ------------------------------------------------------------------------
# 1-6: pure lexical URL features (no network needed)
# ------------------------------------------------------------------------

def having_ip_address(url: str) -> int:
    ip_pattern = (
        r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])|0x[0-9a-fA-F]+"
    )
    return -1 if re.search(ip_pattern, url) else 1


def url_length(url: str) -> int:
    n = len(url)
    if n < 54:
        return 1
    elif n <= 75:
        return 0
    return -1


def shortening_service(url: str) -> int:
    return -1 if re.search(SHORTENERS, url, flags=re.IGNORECASE) else 1


def having_at_symbol(url: str) -> int:
    return -1 if "@" in url else 1


def double_slash_redirecting(url: str) -> int:
    return -1 if url.rfind("//") > 7 else 1


def prefix_suffix(domain: str) -> int:
    return -1 if "-" in domain else 1


def having_sub_domain(netloc: str) -> int:
    host = netloc.split(":")[0]
    host = re.sub(r"^www\.", "", host)
    if _HAS_TLDEXTRACT:
        ext = tldextract.extract(host)
        subdomain = ext.subdomain
        dot_count = subdomain.count(".") + 1 if subdomain else 0
    else:
        registrable = _registrable_domain(netloc)
        remainder = host[: -len(registrable)].rstrip(".") if registrable and host.endswith(registrable) else host
        dot_count = remainder.count(".") + 1 if remainder else 0
    if dot_count == 0:
        return 1
    elif dot_count == 1:
        return 0
    return -1


def https_token(netloc: str) -> int:
    return -1 if "https" in netloc.lower().split(":")[0] else 1


# ------------------------------------------------------------------------
# 8: SSL certificate check (live TLS handshake)
# ------------------------------------------------------------------------

def ssl_final_state(scheme: str, host: str) -> int:
    if scheme != "https":
        return -1
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=REQUEST_TIMEOUT) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
        issuer_parts = dict(x[0] for x in cert.get("issuer", []))
        issuer_org = (issuer_parts.get("organizationName") or issuer_parts.get("commonName") or "").lower()
        not_before = datetime.strptime(cert["notBefore"], "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        age_years = (datetime.now(timezone.utc) - not_before).days / 365.25
        trusted = any(issuer in issuer_org for issuer in TRUSTED_ISSUERS)
        if trusted and age_years >= 1:
            return 1
        elif trusted:
            return 0
        return -1
    except Exception:
        # Couldn't complete the handshake at all (cert error, refused, timeout)
        return -1


# ------------------------------------------------------------------------
# 9: Domain registration length (WHOIS)
# ------------------------------------------------------------------------

def domain_registration_length(whois_info) -> int:
    if whois_info is None:
        return -1
    creation = _first_date(getattr(whois_info, "creation_date", None))
    expiration = _first_date(getattr(whois_info, "expiration_date", None))
    if creation is None or expiration is None:
        return -1
    days_total = (expiration - creation).days
    return -1 if days_total <= 365 else 1


# ------------------------------------------------------------------------
# 10: Favicon
# ------------------------------------------------------------------------

def favicon(soup, page_url: str, page_domain: str) -> int:
    if soup is None:
        return -1
    icon_links = soup.find_all("link", rel=lambda v: v and "icon" in v.lower())
    if not icon_links:
        return 1  # no explicit favicon declared -> not evidence of phishing
    for tag in icon_links:
        href = tag.get("href")
        if not href:
            continue
        full = urljoin(page_url, href)
        icon_domain = _registrable_domain(urlparse(full).netloc)
        if icon_domain and icon_domain != page_domain:
            return -1
    return 1


# ------------------------------------------------------------------------
# 11: Port scan against the table in the paper
# ------------------------------------------------------------------------

def port_check(host: str) -> int:
    for port, should_be_open in PORT_TABLE.items():
        try:
            with socket.create_connection((host, port), timeout=1.5):
                is_open = True
        except Exception:
            is_open = False
        # Flag as phishing only when a port that should be CLOSED is open.
        # (An open 80/443 is expected and fine; we don't penalize a closed
        # 80/443 either, since many hosts simply don't run bare HTTP.)
        if not should_be_open and is_open:
            return -1
    return 1


# ------------------------------------------------------------------------
# 13-15: External-resource ratio features (need parsed HTML)
# ------------------------------------------------------------------------

def _external_ratio(urls, page_url: str, page_domain: str):
    total = 0
    external = 0
    for u in urls:
        if not u or u.startswith("#") or u.lower().startswith("javascript:"):
            continue
        total += 1
        full = urljoin(page_url, u)
        parsed = urlparse(full)
        if not parsed.netloc:
            continue
        res_domain = _registrable_domain(parsed.netloc)
        if res_domain and res_domain != page_domain:
            external += 1
    if total == 0:
        return None
    return external / total


def request_url(soup, page_url: str, page_domain: str) -> int:
    if soup is None:
        return -1
    srcs = [t.get("src") for t in soup.find_all(["img", "script", "audio", "video", "source", "embed"])]
    ratio = _external_ratio(srcs, page_url, page_domain)
    if ratio is None:
        return 1
    pct = ratio * 100
    if pct < 22:
        return 1
    elif pct <= 61:
        return 0
    return -1


def url_of_anchor(soup, page_url: str, page_domain: str) -> int:
    if soup is None:
        return -1
    anchors = soup.find_all("a")
    total = 0
    suspicious_or_external = 0
    for a in anchors:
        href = a.get("href")
        total += 1
        if not href or href.strip() in ("#",) or href.strip().lower().startswith(
            ("javascript:void(0)", "#content", "#skip")
        ):
            suspicious_or_external += 1
            continue
        full = urljoin(page_url, href)
        parsed = urlparse(full)
        if parsed.netloc:
            res_domain = _registrable_domain(parsed.netloc)
            if res_domain and res_domain != page_domain:
                suspicious_or_external += 1
    if total == 0:
        return 1
    pct = suspicious_or_external / total * 100
    if pct < 31:
        return 1
    elif pct <= 67:
        return 0
    return -1


def links_in_tags(soup, page_url: str, page_domain: str) -> int:
    if soup is None:
        return -1
    urls = []
    for tag in soup.find_all("meta"):
        content = tag.get("content")
        if content and content.strip().lower().startswith(("http://", "https://", "//")):
            urls.append(content)
    for tag in soup.find_all("script"):
        if tag.get("src"):
            urls.append(tag.get("src"))
    for tag in soup.find_all("link"):
        if tag.get("href"):
            urls.append(tag.get("href"))
    ratio = _external_ratio(urls, page_url, page_domain)
    if ratio is None:
        return 1
    pct = ratio * 100
    if pct < 17:
        return 1
    elif pct <= 81:
        return 0
    return -1


# ------------------------------------------------------------------------
# 16-17: Forms
# ------------------------------------------------------------------------

def sfh(soup, page_domain: str) -> int:
    if soup is None:
        return -1
    forms = soup.find_all("form")
    if not forms:
        return 1
    worst = 1
    for f in forms:
        action = (f.get("action") or "").strip()
        if action == "" or action.lower() == "about:blank":
            worst = min(worst, -1)
            continue
        full_domain = _registrable_domain(urlparse(action).netloc) if urlparse(action).netloc else None
        if full_domain and full_domain != page_domain:
            worst = min(worst, 0)
    return worst


def submitting_to_email(html: str) -> int:
    if html is None:
        return -1
    return -1 if re.search(r"mailto:|\.mail\s*\(", html, flags=re.IGNORECASE) else 1


# ------------------------------------------------------------------------
# 18: Abnormal URL (WHOIS hostname vs URL)
# ------------------------------------------------------------------------

def abnormal_url(whois_info, domain: str) -> int:
    if whois_info is None:
        return -1
    whois_domain = getattr(whois_info, "domain_name", None)
    if whois_domain is None:
        return -1
    if isinstance(whois_domain, list):
        candidates = [d.lower() for d in whois_domain if d]
    else:
        candidates = [whois_domain.lower()]
    return 1 if domain.lower() in candidates else -1


# ------------------------------------------------------------------------
# 19: Redirect count
# ------------------------------------------------------------------------

def redirect(redirect_count: int) -> int:
    if redirect_count <= 1:
        return 1
    elif redirect_count < 4:
        return 0
    return -1


# ------------------------------------------------------------------------
# 20-23: JS-source pattern checks
# ------------------------------------------------------------------------

def on_mouseover(html: str) -> int:
    if html is None:
        return -1
    return -1 if re.search(r"onmouseover\s*=\s*[\"'][^\"']*window\.status", html, re.IGNORECASE) else 1


def right_click(html: str) -> int:
    if html is None:
        return -1
    return -1 if re.search(r"event\.button\s*==\s*2|oncontextmenu\s*=\s*[\"']return\s*false", html, re.IGNORECASE) else 1


def pop_up_window(html: str) -> int:
    if html is None:
        return -1
    windows = re.findall(r"window\.open\s*\([^)]*\)", html, re.IGNORECASE)
    if not windows:
        return 1
    # Flag only if a popup script coexists with a text input field on the page
    if re.search(r"<input[^>]*type\s*=\s*[\"']?text", html, re.IGNORECASE):
        return -1
    return 1


def iframe(html: str) -> int:
    if html is None:
        return -1
    return -1 if re.search(r"<iframe", html, re.IGNORECASE) else 1


# ------------------------------------------------------------------------
# 24-25: Domain age / DNS
# ------------------------------------------------------------------------

def age_of_domain(whois_info) -> int:
    if whois_info is None:
        return -1
    creation = _first_date(getattr(whois_info, "creation_date", None))
    if creation is None:
        return -1
    age_days = (datetime.now(timezone.utc) - creation).days
    return 1 if age_days >= 180 else -1


def dns_record(domain: str) -> int:
    if _HAS_DNSPYTHON:
        try:
            dns.resolver.resolve(domain, "A")
            return 1
        except Exception:
            return -1
    # Fallback without dnspython: plain socket resolution
    try:
        socket.gethostbyname(domain)
        return 1
    except socket.gaierror:
        return -1


# ------------------------------------------------------------------------
# 26-30: PROXY features -- original data sources are defunct/paywalled.
# See module docstring. Each function documents its substitute and how
# to wire in a real API key if you have one.
# ------------------------------------------------------------------------

def web_traffic(domain: str) -> int:
    """
    PROXY for the paper's Alexa-rank feature (Alexa was shut down in 2022).
    Uses the Tranco list (https://tranco-list.eu), the standard academic
    Alexa replacement, via its public API. Falls back to 'Suspicious' (0)
    if the lookup fails (e.g. no network, rate limited) rather than
    guessing.
    """
    try:
        resp = requests.get(f"https://tranco-list.eu/api/ranks/domain/{domain}", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            ranks = data.get("ranks", [])
            if ranks:
                latest_rank = ranks[-1].get("rank")
                if latest_rank and latest_rank < 100_000:
                    return 1
                elif latest_rank:
                    return 0
        return -1  # not found in Tranco at all -> treat like "no traffic"
    except Exception:
        return 0  # couldn't check -> don't assert phishing, mark suspicious


def page_rank(domain: str) -> int:
    """
    PROXY for Google PageRank (discontinued 2016, no free live replacement).
    If you have an Open PageRank API key (https://www.domcop.com/openpagerank/),
    set OPEN_PAGERANK_API_KEY below and this will use it. Otherwise returns
    0 (Suspicious/unknown) rather than silently defaulting to Phishing.
    """
    OPEN_PAGERANK_API_KEY = None  # <-- put your key here if you have one
    if not OPEN_PAGERANK_API_KEY:
        return 0
    try:
        resp = requests.get(
            "https://openpagerank.com/api/v1.0/getPageRank",
            params={"domains[]": domain},
            headers={"API-OPR": OPEN_PAGERANK_API_KEY},
            timeout=5,
        )
        result = resp.json()["response"][0]
        rank = float(result.get("page_rank_decimal") or 0)
        return -1 if rank < 0.2 else 1
    except Exception:
        return 0


def google_index(domain: str) -> int:
    """
    PROXY: does NOT scrape Google search results (against ToS, unreliable).
    If you have a Google Programmable Search Engine key + CX id, set them
    below to do a real check. Otherwise defaults to 1 (assume indexed),
    matching the paper's implicit default for reachable, resolvable sites.
    """
    GOOGLE_API_KEY = None   # <-- put your key here if you have one
    GOOGLE_CX = None        # <-- and your search engine ID here
    if not (GOOGLE_API_KEY and GOOGLE_CX):
        return 1
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": f"site:{domain}"},
            timeout=5,
        )
        data = resp.json()
        return 1 if int(data.get("searchInformation", {}).get("totalResults", "0")) > 0 else -1
    except Exception:
        return 1


def links_pointing_to_page(domain: str) -> int:
    """
    PROXY: backlink counts require a paid index (Moz/Ahrefs/Majestic).
    No free live source exists. Returns 0 (Suspicious/unknown) unless you
    wire in an API below.
    """
    return 0


def statistical_report(domain: str, ip: str = None) -> int:
    """
    Checks the live PhishTank public database instead of the paper's
    static 2010-2012 top-10 lists. No API key required for basic checks.
    """
    try:
        resp = requests.post(
            "https://checkurl.phishtank.com/checkurl/",
            data={"url": domain, "format": "json"},
            headers=DEFAULT_HEADERS,
            timeout=5,
        )
        data = resp.json()
        in_db = data.get("results", {}).get("in_database", False)
        verified = data.get("results", {}).get("verified", False)
        if in_db and verified:
            return -1
        return 1
    except Exception:
        return 0  # unknown -> don't assert phishing


# ------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------

def extract_features_from_url(url: str, fetch_live: bool = True) -> pd.DataFrame:
    """
    Extract all 30 features for a single URL.

    fetch_live=True (default) performs live HTTP, WHOIS, DNS, and TLS
    lookups -- this is what you want for real feature extraction and will
    take a few seconds per URL. Set fetch_live=False to compute only the
    lexical (URL-string) features instantly, useful for quick tests.
    """
    url = _normalize_url(url)
    parsed = urlparse(url)
    netloc = parsed.netloc
    domain = _registrable_domain(netloc)
    host = netloc.split(":")[0]

    feats = {}

    # --- Lexical, no network needed ---
    feats["having_IP_Address"] = having_ip_address(url)
    feats["URL_Length"] = url_length(url)
    feats["Shortining_Service"] = shortening_service(url)
    feats["having_At_Symbol"] = having_at_symbol(url)
    feats["double_slash_redirecting"] = double_slash_redirecting(url)
    feats["Prefix_Suffix"] = prefix_suffix(netloc)
    feats["having_Sub_Domain"] = having_sub_domain(netloc)
    feats["HTTPS_token"] = https_token(netloc)

    if not fetch_live:
        for col in FEATURES:
            if col not in feats:
                feats[col] = 0
        return pd.DataFrame([feats])[FEATURES]

    # --- Network-dependent lookups (done once, reused across features) ---
    resp, html, redirect_count = _fetch(url)
    soup = BeautifulSoup(html, "html.parser") if html else None
    final_domain = _registrable_domain(urlparse(resp.url).netloc) if resp is not None else domain
    whois_info = _get_whois(domain)

    feats["SSLfinal_State"] = ssl_final_state(parsed.scheme, host)
    feats["Domain_registeration_length"] = domain_registration_length(whois_info)
    feats["Favicon"] = favicon(soup, url, final_domain)
    feats["port"] = port_check(host)
    feats["Request_URL"] = request_url(soup, url, final_domain)
    feats["URL_of_Anchor"] = url_of_anchor(soup, url, final_domain)
    feats["Links_in_tags"] = links_in_tags(soup, url, final_domain)
    feats["SFH"] = sfh(soup, final_domain)
    feats["Submitting_to_email"] = submitting_to_email(html)
    feats["Abnormal_URL"] = abnormal_url(whois_info, domain)
    feats["Redirect"] = redirect(redirect_count)
    feats["on_mouseover"] = on_mouseover(html)
    feats["RightClick"] = right_click(html)
    feats["popUpWidnow"] = pop_up_window(html)
    feats["Iframe"] = iframe(html)
    feats["age_of_domain"] = age_of_domain(whois_info)
    feats["DNSRecord"] = dns_record(domain)
    feats["web_traffic"] = web_traffic(domain)
    feats["Page_Rank"] = page_rank(domain)
    feats["Google_Index"] = google_index(domain)
    feats["Links_pointing_to_page"] = links_pointing_to_page(domain)
    feats["Statistical_report"] = statistical_report(domain)

    return pd.DataFrame([feats])[FEATURES]


def extract_features_batch(urls, fetch_live: bool = True, delay: float = 0.0) -> pd.DataFrame:
    """Run extract_features_from_url over a list of URLs and stack the results."""
    rows = []
    for u in urls:
        try:
            rows.append(extract_features_from_url(u, fetch_live=fetch_live))
        except Exception as e:
            print(f"Failed on {u}: {e}")
        if delay:
            time.sleep(delay)
    if not rows:
        return pd.DataFrame(columns=FEATURES)
    return pd.concat(rows, ignore_index=True)
