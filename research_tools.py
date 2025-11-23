"""
research_tools.py - Company Research Tools (Enhanced)

This module provides functions to fetch and process company information
using the Wikipedia API and mock data fallbacks.

UPDATED:
- Improved fuzzy matching integration
- Better error handling
- Cleaner data extraction
"""

import requests
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from rapidfuzz import process, fuzz
COMPANY_INDICATORS = [
    "company", "corporation", "inc.", "inc ", "llc",
    "ltd", "plc", "subsidiary", "multinational",
    "headquartered", "founded", "enterprise", "manufacturer",
    "producer", "vendor", "industry"
]

NON_COMPANY_TYPES = [
    "politician", "singer", "musician", "actor", "actress",
    "film", "movie", "song", "album", "book", "novel", "fictional",
    "character", "tv series", "season", "episode", "manga", "anime",
    "species", "animal", "bird", "reptile", "insect",
    "bacterium", "virus", "fungus", "plant",
    "river", "lake", "mountain", "village", "town", "city",
    "chemical", "compound", "protein",
    "explicit", "erotic"
]


@dataclass
class ResearchResult:
    """Structured research result."""
    success: bool
    company_name: str
    data: Optional[Dict]
    confidence: float
    sources: List[str]
    gaps: List[str]
    conflicts: List[Dict]
    error: Optional[str] = None


# ============================================
# WIKIPEDIA API INTEGRATION
# ============================================

WIKIPEDIA_REST_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "CompanyResearchAssistant/1.0 (Educational Project; contact@example.com)",
    "Accept": "application/json"
}


def search_wikipedia(query: str, limit: int = 5) -> List[Dict]:
    """Search Wikipedia using the classic API."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "srprop": "snippet|titlesnippet"
    }

    try:
        response = requests.get(WIKIPEDIA_SEARCH_URL, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": clean_html(item.get("snippet", "")),
                "page_id": item.get("pageid")
            })
        return results

    except requests.exceptions.RequestException as e:
        print(f"Wikipedia search error: {e}")
        return []


def get_wikipedia_page(title: str) -> Optional[Dict]:
    """Fetch a clean summary using Wikipedia REST API."""
    formatted = title.replace(" ", "_")
    url = WIKIPEDIA_REST_URL.format(title=formatted)

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "extract" in data and data["extract"]:
            return {
                "title": data.get("title", title),
                "extract": data.get("extract", ""),
                "source": data.get("content_urls", {})
                         .get("desktop", {})
                         .get("page", f"https://en.wikipedia.org/wiki/{formatted}")
            }
        return None

    except requests.exceptions.RequestException as e:
        print(f"Wikipedia fetch error: {e}")
        return None


def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)


# ============================================
# FUZZY MATCHING
# ============================================

def fuzzy_pick_best_match(user_query: str, search_results: List[Dict]) -> Optional[Tuple[Dict, int]]:
    """
    Fuzzy match the user's input against search result titles.
    Returns (best_result_dict, score) or None.
    """
    if not search_results:
        return None

    titles = [item["title"] for item in search_results]
    match = process.extractOne(user_query, titles, scorer=fuzz.WRatio)
    
    if not match:
        return None

    matched_title, score, index = match
    return search_results[index], int(score)


# ============================================
# DATA EXTRACTION
# ============================================

def extract_company_info(wiki_text: str, company_name: str) -> Dict:
    """Extract structured company information from Wikipedia text."""
    # Validate that this is actually a company page
    lower = wiki_text.lower()

# Reject if it clearly describes a non-company entity
    if any(term in lower[:300] for term in NON_COMPANY_TYPES):
        return {}

# Accept only if strong company indicators exist
    indicator_hits = sum(1 for term in COMPANY_INDICATORS if term in lower[:400])

    if indicator_hits < 2:
        return {}   # Not enough evidence it's a company

    info = {
        "description": "",
        "industry": "",
        "products": [],
        "services": [],
        "founded": "",
        "headquarters": "",
        "key_people": [],
        "competitors": [],
        "revenue": "",
        "employees": ""
    }

    if not wiki_text:
        return info

    info["description"] = wiki_text[:1500]

    patterns = {
        "founded": [
            r"founded\s+(?:in\s+)?(\d{4})",
            r"established\s+(?:in\s+)?(\d{4})",
            r"incorporated\s+(?:in\s+)?(\d{4})"
        ],
        "headquarters": [
            r"headquartered\s+in\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|and)",
            r"headquarters\s+(?:is\s+)?(?:in\s+)?([A-Z][a-zA-Z\s,]+?)(?:\.|,|and)",
            r"based\s+in\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|and)"
        ],
        "industry": [
            r"(?:is\s+a[n]?\s+)([A-Za-z\s]+?)\s+company",
            r"(?:is\s+a[n]?\s+)([A-Za-z\s]+?)\s+corporation",
            r"(?:in\s+the\s+)([A-Za-z\s]+?)\s+(?:industry|sector)"
        ],
        "revenue": [
            r"revenue\s+(?:of\s+)?(?:US)?\$?([\d.,]+\s*(?:billion|million|trillion))",
            r"(?:US)?\$?([\d.,]+\s*(?:billion|million|trillion))\s+(?:in\s+)?revenue"
        ],
        "employees": [
            r"([\d,]+)\s+employees",
            r"employs?\s+([\d,]+)\s+(?:people|workers|staff)"
        ]
    }

    text_lower = wiki_text.lower()

    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, wiki_text if field == "headquarters" else text_lower, re.IGNORECASE)
            if match:
                info[field] = match.group(1).strip()
                break

    product_patterns = [
        r"products?\s+(?:include|such as|like)\s+([^.]+)",
        r"(?:known for|famous for)\s+([^.]+)",
        r"services?\s+(?:include|such as|like)\s+([^.]+)"
    ]
    for pattern in product_patterns:
        match = re.search(pattern, text_lower)
        if match:
            products_text = match.group(1)
            info["products"] = [p.strip() for p in re.split(r'[,;]|\band\b', products_text) if p.strip()][:10]
            break

    return info


def identify_data_gaps(info: Dict) -> List[str]:
    """Identify missing or incomplete information."""
    gaps = []
    critical_fields = {
        "description": "Company description",
        "industry": "Industry/sector information",
        "products": "Product/service information"
    }

    for field, name in critical_fields.items():
        if not info.get(field):
            gaps.append(name)

    return gaps


def detect_conflicts(search_results: List[Dict], main_info: Dict) -> List[Dict]:
    """Detect conflicting information across sources."""
    conflicts = []

    if len(search_results) > 1:
        titles = [r.get("title", "") for r in search_results[:5]]
        if len(set(titles)) > 1:
            conflicts.append({
                "type": "ambiguous_name",
                "description": "Multiple companies found with similar names",
                "options": titles
            })

    return conflicts


# ============================================
# MOCK DATA
# ============================================

MOCK_COMPANY_DATA = {
    "apple": {
        "name": "Apple Inc.",
        "description": "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Apple is the world's largest technology company by revenue and one of the world's most valuable companies. It designs, manufactures, and sells consumer electronics, software, and services.",
        "industry": "Technology, Consumer Electronics",
        "founded": "1976",
        "headquarters": "Cupertino, California, USA",
        "products": ["iPhone", "iPad", "Mac", "Apple Watch", "AirPods", "Apple TV"],
        "services": ["App Store", "Apple Music", "iCloud", "Apple TV+", "Apple Pay"],
        "competitors": ["Samsung", "Google", "Microsoft", "Huawei", "Sony"],
        "key_people": ["Tim Cook (CEO)", "Craig Federighi (SVP Software Engineering)"],
        "revenue": "$383 billion (2023)",
        "employees": "164,000+"
    },
    "microsoft": {
        "name": "Microsoft Corporation",
        "description": "Microsoft Corporation is an American multinational technology corporation headquartered in Redmond, Washington. Microsoft is the world's largest software maker by revenue and one of the most valuable companies globally. It develops and sells computer software, consumer electronics, and related services.",
        "industry": "Technology, Software, Cloud Computing",
        "founded": "1975",
        "headquarters": "Redmond, Washington, USA",
        "products": ["Windows", "Microsoft 365", "Xbox", "Surface", "Azure"],
        "services": ["Azure Cloud", "LinkedIn", "GitHub", "Microsoft 365", "Dynamics 365"],
        "competitors": ["Apple", "Google", "Amazon", "Oracle", "Salesforce"],
        "key_people": ["Satya Nadella (CEO)", "Brad Smith (President)"],
        "revenue": "$211 billion (2023)",
        "employees": "221,000+"
    },
    "google": {
        "name": "Google LLC (Alphabet Inc.)",
        "description": "Google LLC is an American multinational technology company focusing on search engine technology, online advertising, cloud computing, computer software, quantum computing, e-commerce, artificial intelligence, and consumer electronics. It is a subsidiary of Alphabet Inc.",
        "industry": "Technology, Internet Services, Advertising",
        "founded": "1998",
        "headquarters": "Mountain View, California, USA",
        "products": ["Google Search", "Chrome", "Android", "Pixel", "Nest"],
        "services": ["Google Cloud", "YouTube", "Google Maps", "Gmail", "Google Workspace"],
        "competitors": ["Microsoft", "Apple", "Amazon", "Meta", "OpenAI"],
        "key_people": ["Sundar Pichai (CEO)"],
        "revenue": "$307 billion (2023)",
        "employees": "182,000+"
    },
    "amazon": {
        "name": "Amazon.com, Inc.",
        "description": "Amazon.com, Inc. is an American multinational technology company focusing on e-commerce, cloud computing, online advertising, digital streaming, and artificial intelligence. It is one of the most valuable companies in the world and one of the largest employers.",
        "industry": "E-commerce, Cloud Computing, Technology",
        "founded": "1994",
        "headquarters": "Seattle, Washington, USA",
        "products": ["Kindle", "Echo", "Fire TV", "Ring"],
        "services": ["Amazon Web Services (AWS)", "Prime Video", "Amazon Prime", "Alexa"],
        "competitors": ["Walmart", "Microsoft", "Google", "Alibaba", "eBay"],
        "key_people": ["Andy Jassy (CEO)", "Jeff Bezos (Founder)"],
        "revenue": "$574 billion (2023)",
        "employees": "1,500,000+"
    },
    "tesla": {
        "name": "Tesla, Inc.",
        "description": "Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas. Tesla designs and manufactures electric vehicles, stationary battery energy storage devices, solar panels, and related products and services.",
        "industry": "Automotive, Clean Energy, Technology",
        "founded": "2003",
        "headquarters": "Austin, Texas, USA",
        "products": ["Model S", "Model 3", "Model X", "Model Y", "Cybertruck", "Powerwall", "Solar Roof"],
        "services": ["Supercharger Network", "Full Self-Driving", "Tesla Insurance"],
        "competitors": ["Ford", "GM", "Volkswagen", "Rivian", "BYD", "Lucid Motors"],
        "key_people": ["Elon Musk (CEO)"],
        "revenue": "$96.7 billion (2023)",
        "employees": "140,000+"
    },
    "meta": {
        "name": "Meta Platforms, Inc.",
        "description": "Meta Platforms, Inc., formerly known as Facebook, Inc., is an American multinational technology conglomerate. The company owns and operates Facebook, Instagram, WhatsApp, and Threads, and is developing virtual and augmented reality products.",
        "industry": "Social Media, Technology, Virtual Reality",
        "founded": "2004",
        "headquarters": "Menlo Park, California, USA",
        "products": ["Facebook", "Instagram", "WhatsApp", "Threads", "Meta Quest", "Ray-Ban Meta"],
        "services": ["Meta Ads", "Workplace", "Horizon Worlds"],
        "competitors": ["Google", "TikTok (ByteDance)", "Snap", "Twitter/X", "Apple"],
        "key_people": ["Mark Zuckerberg (CEO)"],
        "revenue": "$134.9 billion (2023)",
        "employees": "67,000+"
    },
    "infosys": {
        "name": "Infosys Limited",
        "description": "Infosys Limited is an Indian multinational information technology company that provides business consulting, information technology and outsourcing services. Headquartered in Bangalore, Karnataka, India, Infosys is the second-largest Indian IT company after Tata Consultancy Services.",
        "industry": "Information Technology, Business Consulting, Outsourcing",
        "founded": "1981",
        "headquarters": "Bangalore, Karnataka, India",
        "products": ["Infosys Nia", "Infosys Cobalt", "EdgeVerve Systems", "Panaya", "Skava"],
        "services": ["IT Consulting", "Business Process Outsourcing", "Cloud Services", "Digital Transformation", "Application Development"],
        "competitors": ["TCS", "Wipro", "HCL Technologies", "Cognizant", "Accenture", "IBM"],
        "key_people": ["Salil Parekh (CEO)", "Nandan Nilekani (Co-founder)"],
        "revenue": "$18.6 billion (2024)",
        "employees": "317,000+"
    },
    "tcs": {
        "name": "Tata Consultancy Services",
        "description": "Tata Consultancy Services (TCS) is an Indian multinational information technology services and consulting company headquartered in Mumbai. It is a subsidiary of Tata Group and operates in 150 locations across 46 countries. TCS is the largest Indian IT company by market capitalization.",
        "industry": "Information Technology, Consulting, Business Solutions",
        "founded": "1968",
        "headquarters": "Mumbai, Maharashtra, India",
        "products": ["TCS BaNCS", "TCS iON", "ignio", "TCS MasterCraft"],
        "services": ["IT Services", "Consulting", "Business Solutions", "Digital Transformation", "Cloud Infrastructure"],
        "competitors": ["Infosys", "Wipro", "HCL Technologies", "Accenture", "IBM", "Cognizant"],
        "key_people": ["K. Krithivasan (CEO)"],
        "revenue": "$29 billion (2024)",
        "employees": "614,000+"
    },
    "wipro": {
        "name": "Wipro Limited",
        "description": "Wipro Limited is an Indian multinational corporation that provides information technology, consulting and business process services. Headquartered in Bangalore, India, Wipro is one of the leading IT services companies globally.",
        "industry": "Information Technology, Consulting, Business Process Services",
        "founded": "1945",
        "headquarters": "Bangalore, Karnataka, India",
        "products": ["Wipro Holmes", "VIVID", "Wipro Digital Operations Platform"],
        "services": ["Application Services", "Cloud Services", "Consulting", "Digital Operations", "Engineering Services"],
        "competitors": ["TCS", "Infosys", "HCL Technologies", "Cognizant", "Accenture"],
        "key_people": ["Thierry Delaporte (CEO)", "Azim Premji (Founder)"],
        "revenue": "$11.3 billion (2024)",
        "employees": "234,000+"
    },
    "intel": {
        "name": "Intel Corporation",
        "description": "Intel Corporation is an American multinational corporation and technology company headquartered in Santa Clara, California. It is one of the world's largest semiconductor chip manufacturers by revenue.",
        "industry": "Semiconductors, Technology, Computing",
        "founded": "1968",
        "headquarters": "Santa Clara, California, USA",
        "products": ["Intel Core Processors", "Intel Xeon", "Intel Arc GPUs", "Intel Optane"],
        "services": ["Intel Foundry Services", "Intel Developer Cloud"],
        "competitors": ["AMD", "NVIDIA", "Qualcomm", "Samsung", "TSMC", "ARM"],
        "key_people": ["Pat Gelsinger (CEO)"],
        "revenue": "$54.2 billion (2023)",
        "employees": "124,800+"
    },
    "nvidia": {
        "name": "NVIDIA Corporation",
        "description": "NVIDIA Corporation is an American multinational technology company designing and supplying graphics processing units (GPUs), APIs for data science and high-performance computing, as well as system on a chip units for mobile computing and automotive markets.",
        "industry": "Semiconductors, AI, Graphics Processing",
        "founded": "1993",
        "headquarters": "Santa Clara, California, USA",
        "products": ["GeForce GPUs", "RTX Series", "Quadro", "Tesla GPUs", "NVIDIA DGX", "Jetson"],
        "services": ["NVIDIA AI Enterprise", "GeForce NOW", "NVIDIA Omniverse"],
        "competitors": ["AMD", "Intel", "Qualcomm", "Google TPU", "Microsoft"],
        "key_people": ["Jensen Huang (CEO & Co-founder)"],
        "revenue": "$60.9 billion (2024)",
        "employees": "29,600+"
    },
    "netflix": {
        "name": "Netflix, Inc.",
        "description": "Netflix, Inc. is an American subscription video on-demand over-the-top streaming service. Netflix has played a major role in the rise of digital distribution of content.",
        "industry": "Entertainment, Streaming, Technology",
        "founded": "1997",
        "headquarters": "Los Gatos, California, USA",
        "products": ["Netflix Streaming", "Netflix DVD"],
        "services": ["Video Streaming", "Original Content Production"],
        "competitors": ["Disney+", "Amazon Prime Video", "HBO Max", "Apple TV+", "Hulu"],
        "key_people": ["Ted Sarandos (Co-CEO)", "Greg Peters (Co-CEO)"],
        "revenue": "$33.7 billion (2023)",
        "employees": "13,000+"
    }
}


def get_mock_data(company_name: str) -> Optional[Dict]:
    """Get mock data for common companies with fuzzy matching."""
    normalized = company_name.lower().strip()

    # Direct match
    if normalized in MOCK_COMPANY_DATA:
        return MOCK_COMPANY_DATA[normalized]

    # Partial match
    for key, data in MOCK_COMPANY_DATA.items():
        if key in normalized or normalized in key:
            return data
        if normalized in data["name"].lower():
            return data

    # Fuzzy match against mock data keys
    mock_keys = list(MOCK_COMPANY_DATA.keys())
    result = process.extractOne(normalized, mock_keys, scorer=fuzz.WRatio, score_cutoff=80)
    if result:
        matched_key, score, _ = result
        return MOCK_COMPANY_DATA[matched_key]

    return None


# ============================================
# MAIN RESEARCH FUNCTION
# ============================================

def fetch_company_data(company_name: str) -> ResearchResult:
    """
    Main function to fetch company data.
    Tries mock data FIRST for reliability, then Wikipedia API.
    """
    sources = []
    gaps = []
    conflicts = []

    # FIRST: Check mock data
    mock_data = get_mock_data(company_name)

    if mock_data:
        return ResearchResult(
            success=True,
            company_name=mock_data.get("name", company_name),
            data=mock_data,
            confidence=0.95,
            sources=["Internal Database"],
            gaps=identify_data_gaps(mock_data),
            conflicts=[]
        )

    # No mock data - try Wikipedia
    search_query = f"{company_name} company"
    search_results = search_wikipedia(search_query)

    wiki_data = None
    match_score = None

    if search_results:
        conflicts = detect_conflicts(search_results, {})

        # Fuzzy match
        fuzzy = fuzzy_pick_best_match(company_name, search_results)
        if fuzzy:
            best_match, score = fuzzy
            match_score = score
            matched_title = best_match.get("title")
        else:
            best_match = search_results[0]
            matched_title = best_match.get("title")

        wiki_page = get_wikipedia_page(matched_title)

        if wiki_page:
            wiki_data = extract_company_info(wiki_page["extract"], company_name)
            wiki_data["name"] = wiki_page["title"]
            wiki_data["source"] = wiki_page["source"]
            if match_score:
                wiki_data["match_score"] = match_score
            sources.append(wiki_page["source"])

    if wiki_data and wiki_data.get("description"):
        final_data = wiki_data
        confidence = 0.85
        sources.append("Wikipedia")
    else:
    # HARD FAIL — no usable company data found
        return ResearchResult(
        success=False,
        company_name=company_name,
        data=None,
        confidence=0.0,
        sources=[],
        gaps=["No valid company data"],
        conflicts=[],
        error=f"No company found matching '{company_name}'."
    )


    gaps.extend(identify_data_gaps(final_data))

    return ResearchResult(
        success=confidence > 0.3,
        company_name=final_data.get("name", company_name),
        data=final_data,
        confidence=confidence,
        sources=sources,
        gaps=list(set(gaps)),
        conflicts=conflicts
    )


def normalize_research_data(data: Dict) -> Dict:
    """Normalize and clean research data for consistent output."""
    normalized = {
        "name": data.get("name", "Unknown"),
        "description": data.get("description", ""),
        "industry": data.get("industry", ""),
        "founded": data.get("founded", ""),
        "headquarters": data.get("headquarters", ""),
        "products": data.get("products", []),
        "services": data.get("services", []),
        "competitors": data.get("competitors", []),
        "key_people": data.get("key_people", []),
        "revenue": data.get("revenue", ""),
        "employees": data.get("employees", "")
    }

    for field in ["products", "services", "competitors", "key_people"]:
        if isinstance(normalized[field], str):
            normalized[field] = [x.strip() for x in normalized[field].split(",") if x.strip()]

    return normalized


def format_research_for_prompt(result: ResearchResult) -> str:
    """Format research result as a string for LLM prompts."""
    if not result.success:
        return f"Limited information found for {result.company_name}."

    data = result.data
    sections = [
        f"Company: {data.get('name', 'Unknown')}",
        f"Description: {data.get('description', 'N/A')}",
        f"Industry: {data.get('industry', 'N/A')}",
        f"Founded: {data.get('founded', 'N/A')}",
        f"Headquarters: {data.get('headquarters', 'N/A')}",
        f"Products: {', '.join(data.get('products', [])) or 'N/A'}",
        f"Services: {', '.join(data.get('services', [])) or 'N/A'}",
        f"Competitors: {', '.join(data.get('competitors', [])) or 'N/A'}",
        f"Key People: {', '.join(data.get('key_people', [])) or 'N/A'}",
        f"Revenue: {data.get('revenue', 'N/A')}",
        f"Employees: {data.get('employees', 'N/A')}",
        f"\nData Confidence: {result.confidence:.0%}",
        f"Sources: {', '.join(result.sources)}"
    ]

    if result.gaps:
        sections.append(f"Information Gaps: {', '.join(result.gaps)}")

    return "\n".join(sections)