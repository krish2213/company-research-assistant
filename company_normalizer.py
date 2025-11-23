"""

company_normalizer.py - LLM-based Company Name Normalization
"""

import os
import json
import re
from typing import Dict, Optional, Tuple, List
from groq import Groq
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

load_dotenv()
# ============================================
# CLEAN COMMAND PREFIXES (fix for "Research XyzCo")
# ============================================

COMMAND_PREFIXES = [
    "research", "find", "lookup", "search",
    "analyse", "analyze", "investigate",
    "tell me about", "get info on", "get information on"
]

def clean_input(text: str) -> str:
    """Strip command verbs like 'Research Apple' → 'Apple'"""
    lower = text.lower().strip()
    for prefix in COMMAND_PREFIXES:
        if lower.startswith(prefix):
            return text[len(prefix):].strip()
    return text


# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"
 # FIX: Model name updated

# ============================================
# KNOWN COMPANIES DATABASE (for fuzzy matching)
# ============================================

KNOWN_COMPANIES = [
    "Apple", "Microsoft", "Google", "Amazon", "Tesla", "Meta", "Facebook",
    "Netflix", "Adobe", "Salesforce", "Oracle", "IBM", "Intel", "AMD", "NVIDIA",
    "Qualcomm", "Cisco", "Dell", "HP", "Lenovo", "Samsung", "Sony", "LG",
    "Infosys", "TCS", "Wipro", "HCL Technologies", "Cognizant", "Accenture",
    "Twitter", "X", "Snap", "Snapchat", "Pinterest", "LinkedIn", "Uber", "Lyft",
    "Airbnb", "Spotify", "Zoom", "Slack", "Dropbox", "Box", "PayPal", "Stripe",
    "Square", "Block", "Shopify", "eBay", "Walmart", "Target", "Costco",
    "McDonald's", "Starbucks", "Nike", "Adidas", "Coca-Cola", "PepsiCo",
    "Procter & Gamble", "Johnson & Johnson", "Pfizer", "Moderna", "Merck",
    "Boeing", "Airbus", "Ford", "GM", "General Motors", "Toyota", "Honda",
    "Volkswagen", "BMW", "Mercedes-Benz", "Audi", "Rivian", "Lucid Motors",
    "SpaceX", "Blue Origin", "Palantir", "Snowflake", "Databricks", "MongoDB",
    "Atlassian", "ServiceNow", "Workday", "SAP", "VMware", "Red Hat",
    "GitHub", "GitLab", "Docker", "Kubernetes", "OpenAI", "Anthropic",
    "Reliance", "Tata", "Mahindra", "Bharti Airtel", "HDFC", "ICICI",
    "Flipkart", "Zomato", "Swiggy", "Ola", "Paytm", "PhonePe", "Razorpay"
]

# Company aliases and common references
COMPANY_ALIASES = {
    "the search company": "Google",
    "the search engine": "Google",
    "the map company": "Google",
    "the email company": "Google",
    "gmail company": "Google",
    "youtube company": "Google",
    "the iphone company": "Apple",
    "the mac company": "Apple",
    "iphone maker": "Apple",
    "the windows company": "Microsoft",
    "xbox company": "Microsoft",
    "office company": "Microsoft",
    "the cloud company": "Amazon", # <-- Target for this fix
    "aws company": "Amazon",
    "the shopping company": "Amazon",
    "the electric car company": "Tesla",
    "the ev company": "Tesla",
    "elon musk's car company": "Tesla",
    "the social media company": "Meta",
    "facebook company": "Meta",
    "instagram company": "Meta",
    "whatsapp company": "Meta",
    "the indian it company": "Infosys",
    "the indian it firm": "Infosys",
    "bangalore it company": "Infosys",
    "tata it company": "TCS",
    "the gpu company": "NVIDIA",
    "the graphics card company": "NVIDIA",
    "the chip company": "Intel",
    "the processor company": "Intel",
    "the streaming company": "Netflix",
    "the movie streaming": "Netflix",
    "the ride company": "Uber",
    "the taxi app": "Uber",
    "the music app": "Spotify",
    "the music streaming": "Spotify",
}

# Words that should NEVER be treated as company names
NON_COMPANY_WORDS = {
    # Confirmation words
    "yes", "no", "ok", "okay", "sure", "proceed", "continue", "go", "go ahead",
    "fine", "alright", "right", "correct", "yep", "yup", "nope", "nah",
    # Greetings
    "hi", "hello", "hey", "hii", "hiii", "greetings", "good morning", 
    "good afternoon", "good evening", "bye", "goodbye", "thanks", "thank you",
    # Commands
    "help", "stop", "exit", "quit", "cancel", "reset", "clear", "undo",
    "show", "display", "view", "see", "print", "plan",
    # Pronouns and references
    "it", "that", "this", "them", "they", "those", "these",
    "the company", "that company", "this company", "the one",
    # Common words
    "what", "who", "where", "when", "why", "how", "which",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "doing",
    "have", "has", "had", "having",
    "a", "an", "the", "and", "or", "but", "if", "then",
    # Numbers
    "1", "2", "3", "4", "5", "one", "two", "three", "first", "second",
}

# ============================================
# FUZZY MATCHING FUNCTIONS
# ============================================

def fuzzy_match_company(query: str, threshold: int = 70) -> Optional[Tuple[str, int]]:
    """
    Fuzzy match a query against known companies.
    Returns (matched_company, score) or None.
    """
    if not query or len(query) < 2:
        return None
    
    query_lower = query.lower().strip()
    
    # Check exact match first
    for company in KNOWN_COMPANIES:
        if company.lower() == query_lower:
            return (company, 100)
    
    # Fuzzy match against known companies
    result = process.extractOne(
        query, 
        KNOWN_COMPANIES, 
        scorer=fuzz.WRatio,
        score_cutoff=threshold
    )
    
    if result:
        matched, score, _ = result
        return (matched, int(score))
    
    return None


def is_likely_misspelling(query: str) -> bool:
    """
    Check if query looks like a misspelled company name.
    """
    query_lower = query.lower().strip()
    
    # Short single words that look like they could be misspellings
    if len(query_lower) >= 4 and len(query_lower) <= 15:
        # Check if it's close to any known company
        result = fuzzy_match_company(query_lower, threshold=60)
        if result and result[1] >= 60 and result[1] < 100:
            return True
    
    return False


# ============================================
# LLM-BASED COMPANY EXTRACTION
# ============================================

COMPANY_EXTRACTION_PROMPT = """You are a company name extraction assistant. Your job is to analyze user input and determine:
1. Whether they are asking about a specific company
2. What company they mean (correcting misspellings if needed)
3. Your confidence level

IMPORTANT RULES:
- Words like "yes", "no", "ok", "proceed", "continue", "stop", "help" are NEVER company names
- Numbers like "1", "2", "3" are selection responses, not company names
- Phrases like "that company", "the one", "this one" need context to resolve
- Misspellings should be corrected: "micrsooft" → "Microsoft", "gogle" → "Google"
- Contextual hints should be used: "the search engine" → "Google", "iPhone maker" → "Apple"

User's current message: "{user_message}"
Previous context (if any): "{context}"

Respond ONLY with a JSON object (no markdown, no extra text):
{{
    "is_company_query": true/false,
    "extracted_company": "company name or null",
    "corrected_from": "original misspelling or null",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}

Examples:
- "micrsooft" → {{"is_company_query": true, "extracted_company": "Microsoft", "corrected_from": "micrsooft", "confidence": 0.9, "reasoning": "Misspelling of Microsoft"}}
- "yes" → {{"is_company_query": false, "extracted_company": null, "corrected_from": null, "confidence": 1.0, "reasoning": "Confirmation word, not a company"}}
- "the search engine company" → {{"is_company_query": true, "extracted_company": "Google", "corrected_from": null, "confidence": 0.85, "reasoning": "Contextual reference to Google"}}
- "Research Apple" → {{"is_company_query": true, "extracted_company": "Apple", "corrected_from": null, "confidence": 1.0, "reasoning": "Direct company mention"}}
"""

def extract_company_with_llm(user_message: str, context: str = "") -> Dict:
    """
    Use LLM to extract and normalize company name from user input.
    Returns dict with: is_company_query, extracted_company, confidence, etc.
    """
    # Quick pre-checks for obvious non-company inputs
    user_message = clean_input(user_message)

    msg_lower = user_message.lower().strip()
    
    # Check against non-company words
    if msg_lower in NON_COMPANY_WORDS:
        return {
            "is_company_query": False,
            "extracted_company": None,
            "corrected_from": None,
            "confidence": 1.0,
            "reasoning": f"'{user_message}' is a common word, not a company name",
            "is_alias_match": False # NEW: Added flag
        }
    
    # Check for pure numbers (disambiguation responses)
    if re.match(r'^\d+\.?$', msg_lower):
        return {
            "is_company_query": False,
            "extracted_company": None,
            "corrected_from": None,
            "confidence": 1.0,
            "reasoning": "Numeric input - likely a selection response",
            "is_alias_match": False # NEW: Added flag
        }
    
    # === FIX: Prioritize explicit aliases and flag for confirmation ===
    # Check aliases
    for alias, company in COMPANY_ALIASES.items():
        if alias in msg_lower:
            return {
                "is_company_query": True,
                "extracted_company": company,
                "corrected_from": alias, # Using alias as the 'corrected_from' source
                "confidence": 0.65, # Set confidence below 0.9 to force confirmation
                "reasoning": f"Matched contextual alias '{alias}' to {company}",
                "is_alias_match": True # NEW: Added flag
            }
    
    # Try fuzzy matching next (faster than LLM)
    fuzzy_result = fuzzy_match_company(user_message, threshold=75)
    if fuzzy_result:
        matched, score = fuzzy_result
        return {
            "is_company_query": True,
            "extracted_company": matched,
            "corrected_from": user_message if score < 100 else None,
            "confidence": score / 100.0,
            "reasoning": f"Fuzzy matched to {matched} with {score}% confidence",
            "is_alias_match": False # NEW: Added flag
        }
    
    # Call LLM for complex cases
    try:
        prompt = COMPANY_EXTRACTION_PROMPT.format(
            user_message=user_message,
            context=context[:500] if context else "No previous context"
        )
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You extract company names from user input. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean up response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result = json.loads(result_text.strip())
        result["is_alias_match"] = False # NEW: Add default flag
        return result
        
    except Exception as e:
        # Fallback: simple heuristic
        return {
            "is_company_query": len(user_message.split()) <= 3 and user_message[0].isupper(),
            "extracted_company": user_message if len(user_message.split()) <= 3 else None,
            "corrected_from": None,
            "confidence": 0.5,
            "reasoning": f"LLM extraction failed ({str(e)}), using heuristic",
            "is_alias_match": False # NEW: Added flag
        }

def needs_confirmation(extraction_result: Dict) -> bool:
    """
    Determine when to ask user for confirmation.
    """
    if not extraction_result.get("is_company_query"):
        return False

    confidence = extraction_result.get("confidence", 0)
    corrected_from = extraction_result.get("corrected_from")
    is_alias = extraction_result.get("is_alias_match", False)

    # 1. Alias ALWAYS requires confirmation
    if is_alias:
        return True

    # 2. Misspelling corrected → confirmation
    if corrected_from is not None:
        return True

    # 3. Low confidence (< 0.85) → confirmation
    if confidence < 0.85:
        return True

    return False


def format_confirmation_message(extraction_result: Dict) -> str:
    """
    Format a confirmation message for the user.
    """
    company = extraction_result.get("extracted_company")
    corrected_from = extraction_result.get("corrected_from")
    is_alias = extraction_result.get("is_alias_match", False) # NEW: Check for alias match
    
    if is_alias:
        return f"I assume you mean **{company}** (based on your reference to '{corrected_from}').\nPlease reply 'yes' to confirm or provide the correct company name."
    
    if corrected_from:
        return f"Did you mean **{company}**? (I interpreted '{corrected_from}' as {company})\nPlease reply 'yes' to confirm or provide the correct company name."
    else:
        return f"Just to confirm — would you like me to research **{company}**?\nPlease reply 'yes' to confirm or provide a different company name."

# ============================================
# CONTEXT-AWARE RESOLUTION
# ============================================

def resolve_contextual_reference(message: str, conversation_history: List[Dict]) -> Optional[str]:
    """
    Resolve contextual references like "that company", "the one you mentioned".
    """
    msg_lower = message.lower().strip()
    
    # Check for contextual phrases
    contextual_phrases = [
        "that company", "this company", "the company", "the one",
        "that one", "this one", "same company", "same one",
        "the previous one", "the last one", "mentioned company"
    ]
    
    is_contextual = any(phrase in msg_lower for phrase in contextual_phrases)
    
    if not is_contextual:
        return None
    
    # Look for the most recent company mention in history
    for msg in reversed(conversation_history[-10:]):
        content = msg.get("content", "")
        # Check if assistant mentioned a company
        if msg.get("role") == "assistant":
            # Look for company names in the response
            for company in KNOWN_COMPANIES:
                if company.lower() in content.lower():
                    return company
    
    return None


