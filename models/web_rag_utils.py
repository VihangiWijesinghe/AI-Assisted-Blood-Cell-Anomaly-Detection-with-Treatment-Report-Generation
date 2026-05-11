import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from urllib.parse import urlparse


TRUSTED_DOMAINS = [
    "medlineplus.gov",
    "mayoclinic.org",
    "nhs.uk",
    "cdc.gov",
    "nih.gov",
    "nhlbi.nih.gov",
    "bloodcancer.org.uk",
]


FALLBACK_SOURCES = {
    "Leukemia": [
        {
            "title": "MedlinePlus - Leukemia",
            "url": "https://medlineplus.gov/leukemia.html",
            "snippet": "General medical information about leukemia from MedlinePlus."
        },
        {
            "title": "NHLBI - Blood Tests",
            "url": "https://www.nhlbi.nih.gov/health/blood-tests",
            "snippet": "Information about blood tests and complete blood count interpretation from NHLBI."
        },
        {
            "title": "Mayo Clinic - Leukemia",
            "url": "https://www.mayoclinic.org/diseases-conditions/leukemia/symptoms-causes/syc-20374373",
            "snippet": "Overview of leukemia symptoms and causes from Mayo Clinic."
        },
    ],
    "Anemia": [
        {
            "title": "MedlinePlus - Anemia",
            "url": "https://medlineplus.gov/anemia.html",
            "snippet": "General medical information about anemia from MedlinePlus."
        },
        {
            "title": "NHLBI - Anemia",
            "url": "https://www.nhlbi.nih.gov/health/anemia",
            "snippet": "Information about anemia, symptoms, and blood test findings from NHLBI."
        },
        {
            "title": "Mayo Clinic - Anemia",
            "url": "https://www.mayoclinic.org/diseases-conditions/anemia/symptoms-causes/syc-20351360",
            "snippet": "Overview of anemia symptoms and causes from Mayo Clinic."
        },
    ],
    "Infection": [
        {
            "title": "MedlinePlus - White Blood Cell Count",
            "url": "https://medlineplus.gov/lab-tests/white-blood-count-wbc/",
            "snippet": "Information about white blood cell count and what high values may indicate."
        },
        {
            "title": "NHLBI - Blood Tests",
            "url": "https://www.nhlbi.nih.gov/health/blood-tests",
            "snippet": "Information about complete blood count and blood test interpretation from NHLBI."
        },
        {
            "title": "Mayo Clinic - Complete Blood Count",
            "url": "https://www.mayoclinic.org/tests-procedures/complete-blood-count/about/pac-20384919",
            "snippet": "Explanation of complete blood count testing from Mayo Clinic."
        },
    ],
    "Normal": [
        {
            "title": "NHLBI - Blood Tests",
            "url": "https://www.nhlbi.nih.gov/health/blood-tests",
            "snippet": "Information about blood tests and complete blood count interpretation from NHLBI."
        },
        {
            "title": "Mayo Clinic - Complete Blood Count",
            "url": "https://www.mayoclinic.org/tests-procedures/complete-blood-count/about/pac-20384919",
            "snippet": "Explanation of complete blood count testing from Mayo Clinic."
        },
        {
            "title": "MedlinePlus - Complete Blood Count",
            "url": "https://medlineplus.gov/lab-tests/complete-blood-count-cbc/",
            "snippet": "Information about complete blood count testing from MedlinePlus."
        },
    ],
}


def is_trusted_url(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(trusted in domain for trusted in TRUSTED_DOMAINS)


def search_trusted_medical_web(query: str, max_results: int = 5) -> list[dict]:
    results = []

    trusted_sites = [
        "medlineplus.gov",
        "mayoclinic.org",
        "nhs.uk",
        "cdc.gov",
        "nih.gov",
        "nhlbi.nih.gov",
        "bloodcancer.org.uk",
    ]

    try:
        with DDGS() as ddgs:
            for site in trusted_sites:
                search_query = f"{query} site:{site}"

                try:
                    for item in ddgs.text(search_query, max_results=3):
                        url = item.get("href", "")
                        title = item.get("title", "")
                        snippet = item.get("body", "")

                        if url and is_trusted_url(url):
                            results.append({
                                "title": title,
                                "url": url,
                                "snippet": snippet
                            })

                        if len(results) >= max_results:
                            return results

                except Exception:
                    continue

    except Exception:
        return []

    return results


def fetch_page_text(url: str, max_chars: int = 4000) -> str:
    if not is_trusted_url(url):
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())

    return text[:max_chars]


def get_fallback_sources(disease: str, top_k: int = 3) -> list[dict]:
    return FALLBACK_SOURCES.get(disease, FALLBACK_SOURCES["Normal"])[:top_k]


def retrieve_web_evidence(prediction: dict, top_k: int = 3) -> list[dict]:
    disease = prediction.get("predicted_disease") or prediction.get("label", "blood disorder")

    query = f"{disease} complete blood count blood test explanation"

    search_results = search_trusted_medical_web(query, max_results=top_k)

    # Important fix:
    # If DuckDuckGo gives no results, use trusted fallback URLs.
    if not search_results:
        search_results = get_fallback_sources(disease, top_k=top_k)

    evidence = []

    for result in search_results:
        try:
            page_text = fetch_page_text(result["url"])

            evidence.append({
                "title": result.get("title", "Medical Source"),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "content": page_text if page_text else result.get("snippet", "")
            })

        except Exception:
            evidence.append({
                "title": result.get("title", "Medical Source"),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "content": result.get("snippet", "")
            })

    return evidence